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

def init_db():
    conn = sqlite3.connect('detections.db')  # 或你的資料庫路徑
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
    conn.close()

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'animals.pt'))

# 載入 YOLOv5 模型（需安裝 yolov5 repo 並放此 .pt 模型）
import sys
sys.path.append('yolov5')  # 加入 yolov5 的資料夾路徑

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# 初始化 YOLOv5 模型（非 hub 方式）
weights = 'animals.pt'  # 你的模型檔案路徑
device = select_device('')  # 空字串代表自動選 GPU 或 CPU
model = DetectMultiBackend(weights, device=device)
print("✅ 模型載入成功！")

  
# 如果你要使用 YOLO 模型辨識

app = Flask(__name__)

# LINE API 金鑰
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    print("Received body:", body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    if model is None:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="❌ 模型尚未載入，無法辨識圖片。")
        )
        return
    
    message_id = event.message.id

    # 1️⃣ 下載圖片
    image_content = line_bot_api.get_message_content(message_id)
    image_name = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, image_name)

    with open(image_path, 'wb') as f:
        f.write(image_content.content)

   # 2️⃣ 用 YOLOv5 分析圖片（使用 DetectMultiBackend）


    img0 = cv2.imread(image_path)  # 原圖 (BGR)
    img = letterbox(img0, new_shape=640)[0]  # 調整大小
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB → CHW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(model.device)
    img_tensor = img_tensor.float() / 255.0  # 正規化 0~1
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # 增加 batch 維度

    # 模型推論
    pred = model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    # 統計辨識
    detected_items = {}
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img0.shape).round()
        for *box, conf, cls in pred.tolist():
            label = model.names[int(cls)]
            detected_items[label] = detected_items.get(label, 0) + 1

        
    # ✅ 儲存資料進 SQLite（最佳做法）
    DB_PATH = os.path.join(os.path.dirname(__file__), 'detections.db')
    conn = sqlite3.connect(DB_PATH)  # 先建立連線物件
    cursor = conn.cursor()           # 從連線建立 cursor
    time_now_full = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for label, count in detected_items.items():
        cursor.execute('''
        INSERT INTO detections (image_name, timestamp, label, count)
        VALUES (?, ?, ?, ?)
    ''', (image_name, time_now_full, label, count))

    conn.commit()
    conn.close()

    # 4️⃣ 在圖片上畫框
    for *box, conf, cls in pred.tolist():
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        color = (0, 255, 0)  # 綠色框
        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, lineType=cv2.LINE_AA)

    # 儲存結果圖
    result_img_path = os.path.join(UPLOAD_FOLDER, f"result_{image_name}")
    Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)).save(result_img_path)


    # 5️⃣ 組合回傳訊息
    time_now = datetime.datetime.now().strftime('%H:%M')
    message_text = f"辨識時間：{time_now}\n"
    for label, count in detected_items.items():
        message_text += f"{label}: {count}\n"

    # 6️⃣ 準備回傳
    BASE_URL = os.getenv("BASE_URL","https://yolo-line-render.onrender.com")
    image_url = f"{BASE_URL}/static/uploads/result_{image_name}"

    line_bot_api.reply_message(event.reply_token, [
        TextSendMessage(text=message_text),
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

if __name__ == "__main__":
    init_db()  # 啟動伺服器前先確保資料表存在
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
