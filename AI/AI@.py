import discord
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import os
import requests
from datetime import datetime
from discord.ext import commands

# -------------------------------
# ⚙️ CẤU HÌNH
# -------------------------------
TOKEN = "MTM1MTA5MTQzMTkzODQ2MTY5Ng.GhU1-r.i_girg8BP-Np5Hs0JjALiFNOKhGBC4B7Jh5Hz0"
CHANNEL_ID = 1350866408053145672
SAVE_FOLDER = r"E:\PHOTO"
WEBHOOK_URL = "https://discord.com/api/webhooks/1350866530899988500/HYogs0586qBAgtLT9ZYQYrZWk2pZ4aKTSr7HbD6R8nrHdFSUOnyztZzMLxjwO4oagnHO"

# -------------------------------
# 🧠 MÔ HÌNH PYTORCH
# -------------------------------
class WaterColorNet(nn.Module):
    """Phân loại màu nước (đục / trong / hơi đục /...)"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class FlowSpeedNet(nn.Module):
    """Phân loại tốc độ dòng chảy"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load mô hình đã train (nếu có)
device = "cuda" if torch.cuda.is_available() else "cpu"
water_model = WaterColorNet().to(device)
flow_model = FlowSpeedNet().to(device)

# Nếu bạn đã có mô hình train sẵn:
# water_model.load_state_dict(torch.load("water_model.pt", map_location=device))
# flow_model.load_state_dict(torch.load("flow_model.pt", map_location=device))

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor()
])

# -------------------------------
# 🔍 1) PHÂN TÍCH MÀU NƯỚC
# -------------------------------
def predict_water_color(frame):
    x = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = water_model(x)
    pred = torch.argmax(logits, dim=1).item()
    classes = ["Rất đục", "Đục", "Hơi đục", "Khá đục", "Tương đối trong", "Trong"]
    return classes[pred]

# -------------------------------
# 🌊 2) PHÂN TÍCH DÒNG CHẢY
# -------------------------------
def predict_flow_speed(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    if not ret:
        return "Không đọc được video"

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    flow_mags = []

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.resize(mag, (128, 128))
        flow_mags.append(mag)
        gray1 = gray2

    cap.release()
    if not flow_mags:
        return "Không đo được"

    avg_mag = np.mean(flow_mags, axis=0)
    flow_tensor = torch.tensor(np.stack([avg_mag, avg_mag], axis=0)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = flow_model(flow_tensor)
    pred = torch.argmax(logits, dim=1).item()
    classes = ["Dòng chảy bình thường", "Nước chảy mạnh", "Nước chảy xiết"]
    return classes[pred]

# -------------------------------
# ⚖️ 3) ĐÁNH GIÁ TỔNG HỢP
# -------------------------------
def evaluate_stream_quality(turbidity, flow_speed):
    if turbidity in ["Rất đục", "Đục"] and flow_speed in ["Nước chảy xiết", "Nước chảy mạnh"]:
        return "⚠️ Cảnh báo: Tình trạng lũ nghiêm trọng"
    elif turbidity in ["Hơi đục", "Khá đục"] and flow_speed == "Dòng chảy bình thường":
        return "✅ Dòng suối bình thường, cần theo dõi"
    elif turbidity in ["Trong", "Tương đối trong", "Gần trong"] and flow_speed == "Dòng chảy bình thường":
        return "💧 Dòng suối ổn định"
    else:
        return "❓ Kết quả không rõ, cần kiểm tra thêm"

# -------------------------------
# 💬 EMBED GỬI KẾT QUẢ
# -------------------------------
def send_embed(result_text, author_name, author_avatar, image_url):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    embed = {
        "title": "📊 Kết quả phân tích suối (PyTorch)",
        "description": f"{result_text}\n**Thời gian:** {now}",
        "color": 3447003,
        "author": {"name": author_name, "icon_url": author_avatar},
        "image": {"url": image_url},
        "footer": {"text": "AI phân tích tự động", "icon_url": image_url}
    }
    requests.post(WEBHOOK_URL, json={"embeds": [embed]})

# -------------------------------
# 🤖 DISCORD BOT
# -------------------------------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"✅ Bot đã đăng nhập: {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user or message.channel.id != CHANNEL_ID:
        return

    if not message.attachments:
        print("⚠️ Không có tệp video nào từ Raspberry.")
        return

    for file in message.attachments:
        ext = file.filename.split(".")[-1].lower()
        if ext not in ["mp4", "avi", "mov", "mkv"]:
            continue

        save_path = os.path.join(SAVE_FOLDER, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # Tách khung giữa video
        cap = cv2.VideoCapture(save_path)
        mid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        # Phân tích bằng mô hình PyTorch
        turbidity = predict_water_color(frame)
        flow_speed = predict_flow_speed(save_path)
        final_eval = evaluate_stream_quality(turbidity, flow_speed)

        summary = f"🎥 {file.filename}\n🌈 Màu nước: {turbidity}\n💨 Tốc độ dòng chảy: {flow_speed}\n➡️ {final_eval}"

        send_embed(summary, message.author.display_name,
                   message.author.avatar.url if message.author.avatar else "",
                   file.url)

        os.remove(save_path)
        print("✅ Đã gửi kết quả phân tích PyTorch.")

# -------------------------------
# 🚀 CHẠY BOT
# -------------------------------
client.run(TOKEN)
