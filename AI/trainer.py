import os
import time
import random
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
from io import BytesIO
import numpy as np

# ==============================
# Config - chỉnh ở đây nếu cần
# ==============================
DATASET_DIR = "dataset"
COLOR_DIR = os.path.join(DATASET_DIR, "color")
FLOW_DIR = os.path.join(DATASET_DIR, "flow")
MODEL_DIR = "models"
COLOR_MODEL_PATH = os.path.join(MODEL_DIR, "model_color.pth")
FLOW_MODEL_PATH = os.path.join(MODEL_DIR, "model_flow.pth")

COLOR_CLASSES = ["very_turbid", "turbid", "slightly_turbid", "clear"]
FLOW_CLASSES = ["normal_flow", "strong_flow", "very_fast_flow"]

MAX_IMAGES_PER_CLASS = 8          # ảnh cần cho mỗi lớp
TRAIN_INTERVAL_MINUTES = 30       # train mỗi 30 phút
EPOCHS_PER_CYCLE = 10
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Google Custom Search (tùy chọn). Nếu không có, đặt None để dùng Unsplash fallback.
GOOGLE_API_KEY = "AIzaSyAPRLi8H4oq7G6bMd69t6m4dmYh3HN5uMM" 
GOOGLE_CX = "7403d79f8ebe94647"      

# Unsplash simple endpoint (no key required)
UNSPLASH_RANDOM = "https://source.unsplash.com/400x400/?{}"

# ==============================
# Utilities tải ảnh an toàn
# ==============================
def safe_download_image(url, save_path, retries=3, timeout=8):
    """Tải ảnh an toàn với retry, trả True nếu thành công."""
    if url is None:
        return False
    for attempt in range(retries):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code == 200 and r.content:
                try:
                    img = Image.open(BytesIO(r.content)).convert("RGB")
                except Exception:
                    # nếu không mở được, skip
                    return False
                img = img.resize((224, 224))
                img.save(save_path)
                return True
        except Exception:
            pass
        time.sleep(1 + random.random())
    return False

# ==============================
# Google Custom Search helper
# ==============================
def google_image_urls(query, num=8, api_key=None, cx=None):
    """Trả về list url ảnh từ Google Custom Search. Nếu thiếu api_key/cx trả []"""
    if not api_key or not cx:
        return []
    urls = []
    search_url = "https://www.googleapis.com/customsearch/v1"
    # Google 'num' max = 10 per request, use paging if needed
    page = 1
    while len(urls) < num:
        params = {
            "q": query,
            "cx": cx,
            "key": api_key,
            "searchType": "image",
            "num": min(10, num - len(urls)),
            "start": (page - 1) * 10 + 1
        }
        try:
            resp = requests.get(search_url, params=params, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
            items = data.get("items", [])
            for it in items:
                link = it.get("link")
                if link:
                    urls.append(link)
            if not items:
                break
            page += 1
        except Exception:
            break
        time.sleep(0.2 + random.random() * 0.5)
    return urls

# ==============================
# Tải ảnh: ưu tiên Google, fallback Unsplash
# ==============================
def download_for_class(query, save_dir, need_count=8):
    os.makedirs(save_dir, exist_ok=True)
    existing = len([f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if existing >= need_count:
        print(f"[✔] Đã có {existing} ảnh ở {save_dir}, đủ -> bỏ qua tải.")
        return existing

    print(f"[+] Tải ảnh cho '{query}' vào {save_dir} (cần tối đa {need_count})")
    downloaded = existing

    # 1) thử Google nếu có API key
    urls = []
    if GOOGLE_API_KEY and GOOGLE_CX:
        try:
            urls = google_image_urls(query, num=need_count - existing, api_key=GOOGLE_API_KEY, cx=GOOGLE_CX)
            print(f"  → Google trả về {len(urls)} urls")
        except Exception as e:
            print(f"  ⚠️ Google error: {e}")
            urls = []

    # 2) thử các urls từ Google
    for u in urls:
        if downloaded >= need_count:
            break
        fn = os.path.join(save_dir, f"{query.replace(' ', '_')}_{downloaded}.jpg")
        if safe_download_image(u, fn):
            downloaded += 1
        time.sleep(0.4 + random.random() * 0.6)

    # 3) fallback Unsplash (source.unsplash trả ảnh ngẫu nhiên cho mỗi request)
    while downloaded < need_count:
        url = UNSPLASH_RANDOM.format(query.replace(' ', ','))
        fn = os.path.join(save_dir, f"{query.replace(' ', '_')}_unsplash_{downloaded}.jpg")
        if safe_download_image(url, fn):
            downloaded += 1
        else:
            # nếu không tải được, dừng thử vài lần rồi thoát để tránh loop vô tận
            print("  ⚠️ Unsplash tải thất bại, thử lại sau vài giây...")
            time.sleep(2 + random.random() * 3)
            # try again a few times
            # if repeated failure, break to avoid infinite loop (we still may have some images)
            if downloaded == existing:
                # nếu chưa lấy được ảnh nào, cố thử 3 lần rồi thoát
                for _ in range(2):
                    if safe_download_image(url, fn):
                        downloaded += 1
                        break
                    time.sleep(1 + random.random())
                if downloaded == existing:
                    print("  ❌ Không thể tải ảnh từ Unsplash, dừng fetch ở lớp này.")
                    break
        time.sleep(0.4 + random.random() * 0.6)

    print(f"  ✅ Tổng ảnh có trong {save_dir}: {downloaded}")
    return downloaded

# ==============================
# Prepare datasets
# ==============================
def download_color_images():
    for label in COLOR_CLASSES:
        q = f"{label} river water"
        download_for_class(q, os.path.join(COLOR_DIR, label), need_count=MAX_IMAGES_PER_CLASS)

def download_flow_sequences():
    # We simulate flow sequences by downloading multiple frames/images for each label
    for label in FLOW_CLASSES:
        q = f"{label} river water flow"
        download_for_class(q, os.path.join(FLOW_DIR, label), need_count=MAX_IMAGES_PER_CLASS)

# ==============================
# Models
# ==============================
class ColorNet(nn.Module):
    def __init__(self, num_classes=len(COLOR_CLASSES)):
        super().__init__()
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

class FlowNet(nn.Module):
    def __init__(self, num_classes=len(FLOW_CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, 2, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==============================
# Train routines
# ==============================
def train_color_model():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    if not os.path.isdir(COLOR_DIR):
        print("⚠️ COLOR_DIR không tồn tại, bỏ qua train color.")
        return
    # check that there are subdirs with images
    try:
        dataset = datasets.ImageFolder(COLOR_DIR, transform=transform)
    except Exception as e:
        print(f"⚠️ Lỗi khi build ImageFolder cho color: {e}")
        return
    if len(dataset) == 0:
        print("⚠️ Không có ảnh color hợp lệ, bỏ qua train.")
        return

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ColorNet(num_classes=len(dataset.classes)).to(DEVICE)
    if os.path.exists(COLOR_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(COLOR_MODEL_PATH, map_location=DEVICE))
            print("[↻] Đã load model color cũ.")
        except Exception as e:
            print(f"[!] Không thể load model color cũ: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(EPOCHS_PER_CYCLE):
        running_loss, total, correct = 0.0, 0, 0
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        print(f"[COLOR] Epoch {epoch+1}/{EPOCHS_PER_CYCLE} Loss: {running_loss/len(loader):.4f} Acc: {100*correct/total:.2f}%")
    torch.save(model.state_dict(), COLOR_MODEL_PATH)
    print(f"[✓] Saved color model: {COLOR_MODEL_PATH}")

def train_flow_model():
    # build sequences from pairs of images in each class dir
    sequences = []
    labels = []
    for idx, label in enumerate(FLOW_CLASSES):
        label_dir = os.path.join(FLOW_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        for i in range(len(files)-1):
            try:
                p1 = os.path.join(label_dir, files[i])
                p2 = os.path.join(label_dir, files[i+1])
                img1 = Image.open(p1).convert("L").resize((128,128))
                img2 = Image.open(p2).convert("L").resize((128,128))
                a1 = np.array(img1, dtype=np.float32)/255.0
                a2 = np.array(img2, dtype=np.float32)/255.0
                diff = a2 - a1
                seq = np.stack([a1, diff], axis=0)  # shape (2, H, W)
                sequences.append(seq)
                labels.append(idx)
            except Exception:
                continue

    if len(sequences) == 0:
        print("⚠️ Không có dữ liệu flow hợp lệ, bỏ qua train.")
        return

    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = FlowNet(num_classes=len(FLOW_CLASSES)).to(DEVICE)
    if os.path.exists(FLOW_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(FLOW_MODEL_PATH, map_location=DEVICE))
            print("[↻] Đã load model flow cũ.")
        except Exception as e:
            print(f"[!] Không thể load model flow cũ: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(EPOCHS_PER_CYCLE):
        running_loss, total, correct = 0.0, 0, 0
        model.train()
        for seqs, lbls in loader:
            seqs, lbls = seqs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (preds == lbls).sum().item()
        print(f"[FLOW] Epoch {epoch+1}/{EPOCHS_PER_CYCLE} Loss: {running_loss/len(loader):.4f} Acc: {100*correct/total:.2f}%")
    torch.save(model.state_dict(), FLOW_MODEL_PATH)
    print(f"[✓] Saved flow model: {FLOW_MODEL_PATH}")

# ==============================
# Prepare and auto-loop
# ==============================
def prepare_datasets():
    download_color_images()
    download_flow_sequences()

def download_color_images():
    for label in COLOR_CLASSES:
        q = f"{label} river water"
        download_for_class(q, os.path.join(COLOR_DIR, label), MAX_IMAGES_PER_CLASS)

def download_flow_sequences():
    for label in FLOW_CLASSES:
        q = f"{label} river water flow"
        download_for_class(q, os.path.join(FLOW_DIR, label), MAX_IMAGES_PER_CLASS)

def download_for_class(query, save_dir, need_count):
    return download_for_class_impl(query, save_dir, need_count)

def download_for_class_impl(query, save_dir, need_count):
    # wrapper to keep names short
    return download_for_class_worker(query, save_dir, need_count)

def download_for_class_worker(query, save_dir, need_count):
    os.makedirs(save_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if len(existing_files) >= need_count:
        print(f"[✔] {save_dir} already has {len(existing_files)} images.")
        return len(existing_files)

    # try Google if key present
    downloaded = len(existing_files)
    urls = []
    if GOOGLE_API_KEY and GOOGLE_CX:
        try:
            urls = google_image_urls(query, num=need_count - downloaded, api_key=GOOGLE_API_KEY, cx=GOOGLE_CX)
            print(f"  → Google returned {len(urls)} urls for '{query}'")
        except Exception as e:
            print(f"  ⚠️ Google search failed: {e}")
            urls = []

    for u in urls:
        if downloaded >= need_count:
            break
        fname = os.path.join(save_dir, f"{query.replace(' ','_')}_{downloaded}.jpg")
        if safe_download_image(u, fname):
            downloaded += 1
        time.sleep(0.3 + random.random() * 0.7)

    # fallback Unsplash
    attempts = 0
    while downloaded < need_count and attempts < need_count * 3:
        u = UNSPLASH_RANDOM.format(query.replace(' ', ','))
        fname = os.path.join(save_dir, f"{query.replace(' ','_')}_unsplash_{downloaded}.jpg")
        if safe_download_image(u, fname):
            downloaded += 1
        else:
            # if fails, small wait and retry
            time.sleep(1 + random.random())
        attempts += 1
    print(f"  ✅ {downloaded} images present in {save_dir}")
    return downloaded

def auto_train_loop(interval_minutes=30):
    while True:
        print(f"\n=== 🚀 Bắt đầu chu kỳ train mới ({interval_minutes} phút/lần) ===")
        try:
            prepare_datasets()
        except Exception as e:
            print(f"[⚠️] Lỗi chuẩn bị dataset: {e}")

        try:
            train_color_model()
            train_flow_model()
        except Exception as e:
            print(f"[⚠️] Lỗi khi train: {e}")

        print(f"\n[⏳] Nghỉ {interval_minutes} phút trước khi train lại...")
        time.sleep(interval_minutes * 60)

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(COLOR_DIR, exist_ok=True)
    os.makedirs(FLOW_DIR, exist_ok=True)
    auto_train_loop(interval_minutes=TRAIN_INTERVAL_MINUTES)
