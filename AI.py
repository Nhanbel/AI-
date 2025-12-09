import discord
import os
import cv2
import numpy as np
import requests
import asyncio
from discord.ext import commands
from datetime import datetime

# Cấu hình Bot
TOKEN = "MTM1MTA5MTQzMTkzODQ2MTY5Ng.GhU1-r.i_girg8BP-Np5Hs0JjALiFNOKhGBC4B7Jh5Hz0"
CHANNEL_ID = 1350866408053145672
SAVE_FOLDER = r"E:\PHOTO"
WEBHOOK_URL = "https://discord.com/api/webhooks/1350866530899988500/HYogs0586qBAgtLT9ZYQYrZWk2pZ4aKTSr7HbD6R8nrHdFSUOnyztZzMLxjwO4oagnHO"

# Tạo thư mục lưu file nếu chưa có
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
    print(f"Đã tạo thư mục: {SAVE_FOLDER}")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# -------------------------------
# 1) Các hàm phân tích theo phương pháp Hybrid
# -------------------------------

def analyze_water_color_mask(file_path):
    """
    Phân tích độ đục của nước bằng cách:
      1) Tách vùng nước dựa trên mask màu (nâu/đục).
      2) Tính Hue trung bình (và có thể tính thêm Saturation, Value) trên vùng mask.
      3) Phân loại theo nhiều ngưỡng để phản ánh độ đục chính xác hơn.
    """
    image = cv2.imread(file_path)
    if image is None:
        return "Không thể đọc ảnh"

    # Chuyển sang HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tinh chỉnh cho suối Việt Nam (nước nâu đục ~ Hue: 5-35, S: 50-255, V: 50-255)
    # Đây chỉ là gợi ý, bạn cần thử nghiệm thực tế để tìm khoảng tối ưu
    lower_brown = np.array([5, 50, 50])    # Hue=5, Sat=50, Val=50
    upper_brown = np.array([35, 255, 255]) # Hue=35, Sat=255,Val=255

    # Tạo mask chỉ lấy vùng nước nâu
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Lọc nhiễu (morphology)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # xóa đốm nhỏ
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # lấp lỗ hổng

    # Lấy pixel trong vùng nước
    water_pixels = hsv[mask > 0]

    if water_pixels.size == 0:
        # Nếu không tìm thấy vùng nước nâu
        return "Không tìm thấy vùng nước (có thể nước quá trong hoặc quá khác màu)"

    # Tính trung bình Hue (có thể tính thêm sat_mean, val_mean)
    hue_mean = np.mean(water_pixels[:, 0])

    # Ví dụ chia nhiều ngưỡng
    if hue_mean < 20:
        return "Rất đục"
    elif hue_mean < 25:
        return "Đục"
    elif hue_mean < 35:
        return "Hơi đục"
    elif hue_mean < 45:
        return "Khá đục"
    elif hue_mean < 55:
        return "Gần trong"
    elif hue_mean < 65:
        return "Tương đối trong"
    else:
        return "Trong"


def analyze_flow(file_path):
    """
    Phân tích tốc độ dòng chảy trong video sử dụng optical flow Farneback.
    Các ngưỡng dưới đây chỉ là ví dụ và cần hiệu chỉnh dựa trên dữ liệu thực.
    """
    cap = cv2.VideoCapture(file_path)
    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        return "Không thể đọc video"
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    flow_speeds = []
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_speed = np.mean(magnitude)
        flow_speeds.append(avg_speed)
        gray1 = gray2
    cap.release()
    if not flow_speeds:
        return "Không đo được"
    avg_flow_speed = np.mean(flow_speeds)
    if avg_flow_speed < 4.5:
        return "Dòng chảy bình thường"
    elif avg_flow_speed < 8.5:
        return "Nước chảy mạnh"
    else:
        return "Nước chảy xiết"

def evaluate_stream_quality(turbidity, flow_speed):
    """
    Kết hợp kết quả từ phân tích chất lượng nước và tốc độ dòng chảy để đưa ra kết luận.
    Các quy tắc dưới đây là ví dụ và cần được điều chỉnh theo dữ liệu thực.
    """
    if turbidity in ["Rất đục", "Đục"] and flow_speed in ["Nước chảy xiết", "Nước chảy mạnh"]:
        return "Cảnh báo: Tình trạng lũ nghiêm trọng (lũ quét hoặc lũ ông)"
    elif turbidity in ["Hơi đục", "Khá đục"] and flow_speed == "Dòng chảy bình thường":
        return "Dòng suối bình thường, cần theo dõi"
    elif turbidity in ["Trong", "Tương đối trong", "Gần trong"] and flow_speed == "Dòng chảy bình thường":
        return "Dòng suối ổn định"
    else:
        return "Kết quả không rõ, cần kiểm tra thêm"

def analyze_stream(file_path):
    """
    Phương pháp hybrid phân tích file (ảnh hoặc video) của suối.
    Nếu là ảnh: chỉ phân tích chất lượng nước (turbidity).
    Nếu là video: trích xuất khung hình ở giữa để phân tích chất lượng nước,
    và sử dụng toàn bộ video để tính optical flow (flow speed).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        # Nếu là ảnh, chỉ phân tích màu sắc
        turbidity = analyze_water_color_mask(file_path)
        flow_speed = "N/A"
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            return {"error": "Video không có khung hình"}
        # Trích xuất khung hình giữa video để phân tích màu
        middle_idx = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        ret, middle_frame = cap.read()
        if ret:
            temp_image_path = "temp_middle.jpg"
            cv2.imwrite(temp_image_path, middle_frame)
            turbidity = analyze_water_color_mask(temp_image_path)
            os.remove(temp_image_path)
        else:
            turbidity = "Không đo được"
        # Quay lại đầu video và tính optical flow để ước tính tốc độ dòng chảy
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        flow_speed = analyze_flow(file_path)
        cap.release()
    else:
        return {"error": "Định dạng file không được hỗ trợ"}

    final_eval = evaluate_stream_quality(turbidity, flow_speed)
    return {
        "turbidity": turbidity,
        "flow_speed": flow_speed,
        "final_evaluation": final_eval
    }

# -------------------------------
# 2) Hàm hỗ trợ trích xuất thumbnail video & upload ảnh (nếu cần)
# -------------------------------
def extract_thumbnail(video_path):
    """
    Trích xuất khung hình đầu tiên (thumbnail) từ video_path.
    Trả về đường dẫn tới file ảnh thumbnail, hoặc None nếu không thành công.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        thumbnail_path = video_path.rsplit('.', 1)[0] + "_thumb.jpg"
        cv2.imwrite(thumbnail_path, frame)
        return thumbnail_path
    return None

def upload_image(file_path):
    """
    Hàm giả lập upload file_path lên 1 server để lấy URL công khai.
    Hiện tại chỉ trả về 1 URL mặc định (placeholder).
    Bạn có thể thay thế bằng logic upload thật (Imgur API, server riêng, v.v.)
    """
    return "https://your-default-image-url.com/placeholder.jpg"

# -------------------------------
# 3) Hàm gửi embed qua webhook đến server Discord khác
# -------------------------------
def send_embed(combined_result, author_name, author_avatar, image_url):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    embed = {
        "title": "📊 Kết quả phân tích",
        "description": f"{combined_result}\n**Thời gian:** {current_time}",
        "color": 3447003,
        "author": {
            "name": author_name,
            "icon_url": author_avatar
        },
        "thumbnail": {
            "url": "https://student.husc.edu.vn/Themes/Login/images/Logo-ko-nen.png"
        },
        "image": {
            "url": image_url
        },
        "footer": {
            "text": "Hệ thống phân tích tự động",
            "icon_url": "https://student.husc.edu.vn/Themes/Login/images/Logo-ko-nen.png"
        }
    }
    payload = {"embeds": [embed]}
    response = requests.post(WEBHOOK_URL, json=payload)
    if response.status_code == 204:
        print("🎉 Gửi embed qua webhook thành công!")
    else:
        print(f"❌ Lỗi khi gửi webhook: {response.status_code} - {response.text}")

# -------------------------------
# 4) Sự kiện on_ready và on_message
# -------------------------------
@client.event
async def on_ready():
    print(f"Bot đã đăng nhập với tên: {client.user}")
    print("Bot đang chờ tin nhắn mới...")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.channel.id == CHANNEL_ID:
        print(f"Nhận được tin nhắn từ {message.author}: {message.content}")
        combined_result = ""
        big_image_url = ""  # URL để hiển thị ảnh lớn trong embed

        # Nếu có file đính kèm
        if message.attachments:
            for attachment in message.attachments:
                file_name = attachment.filename
                lower_file = file_name.lower()
                save_path = os.path.join(SAVE_FOLDER, file_name)
                try:
                    data = await attachment.read()
                    with open(save_path, "wb") as f:
                        f.write(data)
                    print(f"Đã tải file: {save_path}")
                except Exception as e:
                    print(f"Lỗi khi tải file {file_name}: {e}")
                    continue

                # Phân tích Hybrid
                analysis = analyze_stream(save_path)
                if "error" in analysis:
                    combined_result += f"File {file_name} không được hỗ trợ hoặc video lỗi\n"
                    big_image_url = "https://your-default-image-url.com/placeholder.jpg"
                else:
                    turbidity = analysis["turbidity"]
                    flow_speed = analysis["flow_speed"]
                    final_eval = analysis["final_evaluation"]

                    # Ghép kết quả
                    if lower_file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        combined_result += f"Ảnh {file_name}: {turbidity}\n"
                        big_image_url = attachment.url
                    elif lower_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        combined_result += f"Video {file_name}: {turbidity}, {flow_speed}\n"
                        # Lấy thumbnail
                        thumb_path = extract_thumbnail(save_path)
                        if thumb_path:
                            big_image_url = upload_image(thumb_path)
                        else:
                            big_image_url = "https://your-default-image-url.com/placeholder.jpg"

                    # Thêm dòng kết luận
                    combined_result += f"=> {final_eval}\n"

            combined_result = combined_result.strip()
            print("Kết quả phân tích:", combined_result)

            # Gửi embed qua webhook đến server Discord khác
            send_embed(
                combined_result,
                message.author.display_name,
                message.author.avatar.url if message.author.avatar else "",
                big_image_url
            )
        else:
            print("Không tìm thấy file đính kèm trong tin nhắn.")

# Khởi chạy bot
client.run(TOKEN)
