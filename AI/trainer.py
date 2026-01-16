#!/usr/bin/env python3
"""
train_pipeline_pixabay.py

Tự động: dùng Pixabay API để tải video suối (nội dung train), chuẩn hóa clips 5s@720p,
trích frames, tính optical flow, gán nhãn tự động theo magnitude → tạo pairs → train FlowNet & ColorNet.
Chạy lặp: train xong nghỉ 1 phút → lặp lại.

Yêu cầu hệ thống:
 - ffmpeg (trong PATH)
 - Python packages: pip install torch torchvision opencv-python pillow requests numpy
 - (tùy chọn) Pixabay API key (đặt biến PIXABAY_API_KEY)

Sử dụng: chỉnh PIXABAY_API_KEY ở phần CONFIG nếu có, sau đó chạy:
    python train_pipeline_pixabay.py

File này tự động lưu models/ model_color.pth và model_flow.pth
"""

import os
import time
import random
import subprocess
import glob
from pathlib import Path
import json

import requests
import numpy as np
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================
# CONFIG
# ==============================
BASE = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
RAW_CLIP_DIR = os.path.join(DATASET_DIR, "raw_clips")   # downloaded clips (unlabeled)
FLOW_CLIP_DIR = os.path.join(DATASET_DIR, "flow_clips") # normalized 5s clips
FLOW_FRAMES_DIR = os.path.join(DATASET_DIR, "flow_frames")
COLOR_DIR = os.path.join(DATASET_DIR, "color")
MODEL_DIR = os.path.join(BASE, "models")
COLOR_MODEL_PATH = os.path.join(MODEL_DIR, "model_color.pth")
FLOW_MODEL_PATH = os.path.join(MODEL_DIR, "model_flow.pth")

COLOR_CLASSES = ["very_turbid", "turbid", "slightly_turbid", "clear"]
FLOW_CLASSES = ["normal_flow", "strong_flow", "very_fast_flow"]

PIXABAY_API_KEY = "54227371-5c9a572e91653f290832af7fa"  
PIXABAY_PER_PAGE = 50
PIXABAY_VIDEO_QUERIES = [
    "river stream flowing",
    "mountain stream",
    "river rapids",
    "flood river",
    "fast river flow",
]

# mapping query -> label auto (script will auto-label by magnitude later)
# number of clips to download per query
CLIPS_PER_QUERY = 6

# clip/frame params
CLIP_DURATION = 5
EXTRACT_FPS = 10
FRAME_SIZE_FLOW = (128, 128)
FRAME_SIZE_COLOR = (224, 224)

# train params
BATCH_SIZE = 8
EPOCHS = 8
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# misc
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MAX_WORKERS = min(8, (os.cpu_count() or 2) * 2)
FFMPEG = "ffmpeg"

# ==============================
# Utilities
# ==============================

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(COLOR_DIR, exist_ok=True)
    os.makedirs(RAW_CLIP_DIR, exist_ok=True)
    os.makedirs(FLOW_CLIP_DIR, exist_ok=True)
    os.makedirs(FLOW_FRAMES_DIR, exist_ok=True)


def is_tool(name):
    import shutil
    return shutil.which(name) is not None


def safe_save_state(state_dict, path):
    tmp = path + ".tmp"
    torch.save(state_dict, tmp)
    os.replace(tmp, path)
    size = os.path.getsize(path)
    if size == 0:
        raise RuntimeError("Saved file size = 0")
    print(f"[✓] saved {path} ({size} bytes)")

# ==============================
# Pixabay downloader
# ==============================

def pixabay_search_videos(query, per_page=PIXABAY_PER_PAGE, api_key=None):
    if api_key is None:
        api_key = os.environ.get('PIXABAY_API_KEY') or PIXABAY_API_KEY
    if not api_key:
        print("[DL] Pixabay API key not set; skip pixabay search")
        return []
    url = "https://pixabay.com/api/videos/"
    params = {"key": api_key, "q": query, "per_page": per_page}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get('hits', [])
    except Exception as e:
        print("[DL] Pixabay request failed:", e)
        return []


def download_video_url(url, dest_path, session=None, timeout=30):
    session = session or requests
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = dest_path + '.tmp'
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1<<20):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, dest_path)
            return True
    except Exception as e:
        if os.path.exists(dest_path + '.tmp'):
            try:
                os.remove(dest_path + '.tmp')
            except Exception:
                pass
        print(f"[DL] download failed {url}: {e}")
        return False


def download_clips_from_pixabay(queries, per_query=CLIPS_PER_QUERY):
    os.makedirs(RAW_CLIP_DIR, exist_ok=True)
    session = requests.Session()
    downloaded = []
    for q in queries:
        hits = pixabay_search_videos(q, per_page=per_query*2)
        if not hits:
            continue
        count = 0
        for h in hits:
            # choose best available resolution (large -> medium -> small)
            vids = h.get('videos', {})
            url = None
            for k in ('large','medium','small'):
                if k in vids:
                    url = vids[k].get('url')
                    break
            if not url:
                continue
            fname = os.path.basename(url.split('?')[0])
            dest = os.path.join(RAW_CLIP_DIR, fname)
            if os.path.exists(dest):
                print(f"[DL] exists {dest}")
                downloaded.append(dest)
                count += 1
            else:
                ok = download_video_url(url, dest, session=session)
                if ok:
                    downloaded.append(dest)
                    count += 1
            if count >= per_query:
                break
    print(f"[DL] downloaded {len(downloaded)} raw clips")
    return downloaded

# ==============================
# Clip normalize: cut 5s@720p
# ==============================

def clip_to_5s(src_path, out_dir=FLOW_CLIP_DIR, clip_duration=CLIP_DURATION):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_path))[0]
    dest = os.path.join(out_dir, f"{base}_clip.mp4")
    if os.path.exists(dest):
        return dest
    if not is_tool(FFMPEG):
        # Try common Windows ffmpeg.exe fallback
        if is_tool('ffmpeg.exe'):
            ffmpeg_exec = 'ffmpeg.exe'
        else:
            print("[CLIP] ffmpeg not available; please install ffmpeg and add to PATH")
            return None
    else:
        ffmpeg_exec = FFMPEG
    cmd = [ffmpeg_exec, '-y', '-ss', '0', '-t', str(clip_duration), '-i', src_path,
           '-vf', 'scale=1280:720', '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', dest]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=(os.name=='nt'))
        return dest
    except Exception as e:
        print(f"[CLIP] ffmpeg failed for {src_path}: {e}")
        return None

# ==============================
# Extract frames and compute mean flow magnitude for labeling
# ==============================

def extract_frames(clip_path, out_dir, fps=EXTRACT_FPS, size=FRAME_SIZE_FLOW, force=False):
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(glob.glob(os.path.join(out_dir, '*.jpg')))
    if frames and not force:
        return frames
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return []
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(video_fps / fps)))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            f = cv2.resize(frame, size)
            fname = os.path.join(out_dir, f"{saved:04d}.jpg")
            cv2.imwrite(fname, f)
            saved += 1
        idx += 1
    cap.release()
    return sorted(glob.glob(os.path.join(out_dir, '*.jpg')))


def mean_flow_for_clip(clip_path):
    temp_dir = os.path.join(FLOW_FRAMES_DIR, '_tmp', os.path.splitext(os.path.basename(clip_path))[0])
    frames = extract_frames(clip_path, temp_dir)
    if len(frames) < 2:
        return 0.0
    mags = []
    prev = cv2.imread(frames[0], cv2.IMREAD_GRAYSCALE)
    for f in frames[1:]:
        cur = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if cur is None or prev is None:
            continue
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
        mags.append(mag)
        prev = cur
    return float(np.mean(mags)) if mags else 0.0

# ==============================
# Labeling clips by tertiles
# ==============================

def label_clips(clips):
    if not clips:
        return {}
    print(f"[LABEL] computing mean flow for {len(clips)} clips...")
    clip2mag = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs = {exe.submit(mean_flow_for_clip, c): c for c in clips}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                m = fut.result()
            except Exception:
                m = 0.0
            clip2mag[c] = m
    mags = list(clip2mag.values())
    if not mags:
        return {}
    p33 = np.percentile(mags, 33)
    p66 = np.percentile(mags, 66)
    mapping = {}
    for c, m in clip2mag.items():
        if m <= p33:
            lab = 0
        elif m <= p66:
            lab = 1
        else:
            lab = 2
        mapping[c] = {'mag': m, 'label': lab}
    print(f"[LABEL] thresholds: {p33:.4f}, {p66:.4f}; counts:",
          [sum(1 for v in mapping.values() if v['label']==i) for i in range(3)])
    return mapping

# ==============================
# Build pairs from labeled clips
# ==============================

def build_pairs(labeled_clips, per_clip_limit=None):
    pairs = []
    for clip_path, info in labeled_clips.items():
        label = info['label']
        clip_id = os.path.splitext(os.path.basename(clip_path))[0]
        frames_out = os.path.join(FLOW_FRAMES_DIR, clip_id)
        frames = extract_frames(clip_path, frames_out)
        for i in range(len(frames)-1):
            pairs.append((frames[i], frames[i+1], label))
            if per_clip_limit and len(pairs) >= per_clip_limit:
                break
    print(f"[PAIRS] built {len(pairs)} pairs")
    return pairs

# ==============================
# Dataset & model
# ==============================
class FlowPairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        p1,p2,label = self.pairs[idx]
        i1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        i2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        if i1 is None or i2 is None:
            flow = np.zeros((2, FRAME_SIZE_FLOW[1], FRAME_SIZE_FLOW[0]), dtype=np.float32)
        else:
            if (i1.shape[1], i1.shape[0]) != FRAME_SIZE_FLOW:
                i1 = cv2.resize(i1, FRAME_SIZE_FLOW)
                i2 = cv2.resize(i2, FRAME_SIZE_FLOW)
            flow = cv2.calcOpticalFlowFarneback(i1, i2, None,
                                               pyr_scale=0.5, levels=3, winsize=15,
                                               iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            clip = np.percentile(np.abs(flow), 98)
            if clip <= 0:
                clip = 1.0
            flow = np.clip(flow, -clip, clip) / clip
            flow = np.transpose(flow, (2,0,1)).astype(np.float32)
        return torch.from_numpy(flow), int(label)

class FlowNetV2(nn.Module):
    def __init__(self, num_classes=len(FLOW_CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ColorNet
class ColorNet(nn.Module):
    def __init__(self, num_classes=len(COLOR_CLASSES)):
        super().__init__()
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

# ==============================
# Train functions
# ==============================

def train_color_model():
    transform = transforms.Compose([
        transforms.Resize(FRAME_SIZE_COLOR),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    if not os.path.isdir(COLOR_DIR):
        print("[COLOR] no color dir, skip")
        return
    dataset = datasets.ImageFolder(COLOR_DIR, transform=transform)
    if len(dataset) == 0:
        print("[COLOR] empty dataset, skip")
        return
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = ColorNet(num_classes=len(dataset.classes)).to(DEVICE)
    if os.path.exists(COLOR_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(COLOR_MODEL_PATH, map_location=DEVICE))
            print("[COLOR] loaded existing model")
        except Exception:
            pass
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    for e in range(EPOCHS):
        running_loss, total, correct = 0.0, 0, 0
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            _, preds = torch.max(out,1)
            total += labels.size(0)
            correct += (preds==labels).sum().item()
        print(f"[COLOR] Epoch {e+1}/{EPOCHS} Loss: {running_loss/len(loader):.4f} Acc: {100*correct/total:.2f}%")
    safe_save_state(model.state_dict(), COLOR_MODEL_PATH)


def train_flow_model(pairs):
    if len(pairs) == 0:
        print("[FLOW] no pairs, skip")
        return
    random.shuffle(pairs)
    dataset = FlowPairsDataset(pairs)
    labels = [p[2] for p in pairs]
    counts = np.bincount(labels, minlength=len(FLOW_CLASSES)).astype(float)
    class_weights = 1.0 / (counts + 1e-8)
    sample_weights = [class_weights[l] for l in labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    model = FlowNetV2(num_classes=len(FLOW_CLASSES)).to(DEVICE)
    if os.path.exists(FLOW_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(FLOW_MODEL_PATH, map_location=DEVICE))
            print("[FLOW] loaded existing model")
        except Exception:
            pass
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    for e in range(EPOCHS):
        running_loss, total, correct = 0.0, 0, 0
        model.train()
        for seqs, lbls in loader:
            seqs, lbls = seqs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            out = model(seqs)
            loss = crit(out, lbls)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            _, preds = torch.max(out,1)
            total += lbls.size(0)
            correct += (preds==lbls).sum().item()
        acc = 100*correct/total if total>0 else 0.0
        print(f"[FLOW] Epoch {e+1}/{EPOCHS} Loss: {running_loss/len(loader):.4f} Acc: {acc:.2f}%")
    safe_save_state(model.state_dict(), FLOW_MODEL_PATH)

# ==============================
# Orchestrator: download -> clip -> label -> pairs -> train -> sleep
# ==============================

def extract_and_save_color_frames(labeled_clips, color_dir=COLOR_DIR, fps=2, size=FRAME_SIZE_COLOR):
    """
    Tách frame từ video đã gán nhãn flow, lưu vào thư mục color/<label>/.
    """
    import shutil
    # Xóa thư mục color cũ để tránh trộn nhãn cũ
    if os.path.exists(color_dir):
        shutil.rmtree(color_dir)
    os.makedirs(color_dir, exist_ok=True)
    for clip_path, info in labeled_clips.items():
        label = info['label']
        label_name = FLOW_CLASSES[label]  # dùng nhãn flow làm nhãn color
        out_dir = os.path.join(color_dir, label_name)
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            continue
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, int(round(video_fps / fps)))
        idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                f = cv2.resize(frame, size)
                fname = os.path.join(out_dir, f"{Path(clip_path).stem}_{saved:04d}.jpg")
                cv2.imwrite(fname, f)
                saved += 1
            idx += 1
        cap.release()
    print("[COLOR] Extracted frames for color training.")

def remove_downloaded_videos():
    """Xoá toàn bộ video đã tải về trong RAW_CLIP_DIR và FLOW_CLIP_DIR."""
    for folder in [RAW_CLIP_DIR, FLOW_CLIP_DIR, FLOW_FRAMES_DIR]:
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, "*"))
            for f in files:
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                    elif os.path.isdir(f):
                        import shutil
                        shutil.rmtree(f)
                except Exception as e:
                    print(f"[CLEAN] Failed to remove {f}: {e}")

def pipeline_once():
    ensure_dirs()
    # 1) download from pixabay
    clips = []
    if PIXABAY_API_KEY or os.environ.get('PIXABAY_API_KEY'):
        print("[DL] harvesting from Pixabay...")
        clips = download_clips_from_pixabay(PIXABAY_VIDEO_QUERIES, per_query=CLIPS_PER_QUERY)
    else:
        print("[DL] No Pixabay key; using existing clips in", RAW_CLIP_DIR)
        clips = sorted(glob.glob(os.path.join(RAW_CLIP_DIR, "*.mp4")))

    # 2) clip to 5s
    normalized = []
    for c in clips:
        dest = clip_to_5s(c)
        if dest:
            normalized.append(dest)
    # also include pre-existing clips in FLOW_CLIP_DIR
    existing = sorted(glob.glob(os.path.join(FLOW_CLIP_DIR, "*_clip.mp4")))
    for e in existing:
        if e not in normalized:
            normalized.append(e)

    if not normalized:
        print("[DL] No clips available after normalization; skipping flow training this round")
        return

    # 3) label by magnitude
    mapping = label_clips(normalized)
    # 4) build pairs
    pairs = build_pairs(mapping)
    # 5) train flow
    train_flow_model(pairs)

    # 6) extract frames for color training (từ video đã gán nhãn flow)
    extract_and_save_color_frames(mapping)

    # 7) train color model trên ảnh vừa tách
    train_color_model()

    # 8) Xoá toàn bộ video và dữ liệu tạm để chuẩn bị cho lô mới
    remove_downloaded_videos()
    print("[CLEAN] Đã xoá toàn bộ video và dữ liệu tạm, sẵn sàng cho lô mới.")

# ==============================
# Loop
# ==============================

def main_loop(interval_minutes=1):
    while True:
        print('=== start cycle ===')
        try:
            pipeline_once()
        except Exception as e:
            print('[ERR] pipeline exception:', e)
        print(f"=== cycle done. sleeping {interval_minutes} minute(s) ===")
        time.sleep(interval_minutes * 60)

if __name__ == '__main__':
    ensure_dirs()
    # allow PIXABAY_API_KEY from env if set
    if not PIXABAY_API_KEY:
        PIXABAY_API_KEY = os.environ.get('PIXABAY_API_KEY')
    main_loop(interval_minutes=1)
