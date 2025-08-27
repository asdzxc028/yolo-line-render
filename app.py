from flask import Flask, request, abort, send_from_directory
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, ImageSendMessage
from linebot.exceptions import InvalidSignatureError
import numpy as np
import datetime
import os
from PIL import Image
import uuid
import cv2
import torch
import sqlite3

# 載入 YOLOv5 模型（需安裝 yolov5 repo 並放此 .pt 模型）
import sys
sys.path.append('yolov5')  # 加入 yolov5 的資料夾路徑

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from utils.general import non_max_suppression  # 不要匯入 scale_coords

# 👉 自己補上 scale_coords 函數
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """將預測座標從模型圖像尺寸映射回原始圖像尺寸"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

# 初始化 YOLOv5 模型（非 hub 方式）
weights = 'animals.pt'  # 你的模型檔案路徑
  
# 如果你要使用 YOLO 模型辨識

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DB_PATH = os.path.join(UPLOAD_FOLDER, 'detections.db')
device = select_device('')
weights_path = os.getenv("MODEL_PATH", "animals.pt")
model = DetectMultiBackend(weights_path, device=device)
model.names = model.names or {i: f'class_{i}' for i in range(1000)}

def init_db():
    with sqlite3.connect(DB_PATH) as conn:  # 或你的資料庫路徑
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            timestamp TEXT,
            label TEXT,
            count INTEGER
           )
        ''')
        conn.commit()
    
# LINE API 金鑰
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    print(f"🔽 收到 webhook 請求: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id 
    detected_items = {} # 防止模型推論錯誤時報錯
    image_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, image_name)

    # 1️⃣ 下載圖片
    try:
        message_id = event.message.id     
        image_content = line_bot_api.get_message_content(message_id)
    except Exception as e:
        print(f"❌ 圖片下載失敗: {e}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 圖片下載失敗"))
        return
    # 2️⃣ 儲存圖片   
    try:    
        with open(image_path, 'wb') as f:
            for chunk in image_content.iter_content():
                f.write(chunk)
                print(f"📩 收到圖片 ID: {message_id}")
                print(f"🔽 下載圖片成功：{image_path}")
    except Exception as e:
        print(f"❌ 圖片儲存失敗: {e}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 圖片儲存失敗"))
        return
    
    # 3️⃣ 圖片讀取
    img0 = cv2.imread(image_path)
    if img0 is None:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="❌ 圖片讀取失敗，請確認是否為有效的圖片檔案。")
        )
        return

    # 4️⃣ 處理圖片
    img = letterbox(img0, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB → CHW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(model.device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = None  # 初始化 pred，避免 finally 中引用未定義
    # 5️⃣ 推論
    try:
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img0.shape).round()
            for *box, conf, cls in pred.tolist():
                label = model.names[int(cls)]
                detected_items[label] = detected_items.get(label, 0) + 1
    except Exception as e:
        print(f"❌ 模型推論失敗: {e}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="❌ 模型辨識失敗，請稍後再試。")
        )
        return
    finally:
       del img_tensor
       torch.cuda.empty_cache()
    # 6️⃣ 儲存到資料庫
    time_now_full = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for label, count in detected_items.items():
            cursor.execute('''
            INSERT INTO detections (image_name, timestamp, label, count)
            VALUES (?, ?, ?, ?)
        ''', (image_name, time_now_full, label, count))
        conn.commit()

    # 7️⃣ 在圖片上畫框
    for *box, conf, cls in pred.tolist():
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        color = (0, 255, 0)
        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, lineType=cv2.LINE_AA)

    result_img_path = os.path.join(UPLOAD_FOLDER, f"result_{image_name}")
    Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)).save(result_img_path)
    print(f"🖼️ 標註圖已儲存：{result_img_path}")

    # 8️⃣ 回傳 LINE 訊息
    time_now = datetime.datetime.now().strftime('%H:%M')
    message_text = f"辨識時間：{time_now}\n"
    for label, count in detected_items.items():
        message_text += f"{label}: {count}\n"

    BASE_URL = os.getenv("BASE_URL", "https://yolo-line-render.onrender.com")
    image_url = f"{BASE_URL}/static/uploads/result_{image_name}"    
    db_download_url = f"{BASE_URL}/download-db"

    line_bot_api.reply_message(event.reply_token, [
        TextSendMessage(text=message_text),
        TextSendMessage(text=f"📥 下載資料庫檔案：{db_download_url}"),
        ImageSendMessage(
            original_content_url=image_url,
            preview_image_url=image_url
        )
    ])

# 讓 Render 能讀圖片網址
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/')
def home():
    return "Flask is running."

@app.route('/download-db')
def download_db():
    max_size = 500 * 1024 * 1024  # 500 MB
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) <= max_size:
        return send_from_directory(UPLOAD_FOLDER, 'detections.db', as_attachment=True)
    else:
        return "❌ 資料庫檔案過大，無法提供下載", 413

@app.route('/clear-db')
def clear_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM detections')
        conn.commit()
    return "✅ 資料庫已清空"

if __name__ == "__main__":
    init_db()  # 啟動伺服器前先確保資料表存在
    port = int(os.environ.get('PORT', 5000))  # Render 會給你正確的 port
    app.run(host='0.0.0.0', port=port)
