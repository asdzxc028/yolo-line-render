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
    # 1️⃣ 延遲載入 YOLOv5 模型
    try:
        device = select_device('')
        model = DetectMultiBackend('animals.pt', device=device)
        model.names = model.names or {i: f'class_{i}' for i in range(1000)}  # 預設 class 名稱
    except Exception as e:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"❌ 模型載入失敗：{e}")
        )
        return

    message_id = event.message.id

    # 2️⃣ 下載圖片
    image_content = line_bot_api.get_message_content(message_id)
    image_name = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, image_name)

    with open(image_path, 'wb') as f:
        f.write(image_content.content)

    # 3️⃣ 處理圖片與推論
    img0 = cv2.imread(image_path)
    img = letterbox(img0, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB → CHW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(model.device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    detected_items = {}
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img0.shape).round()
        for *box, conf, cls in pred.tolist():
            label = model.names[int(cls)]
            detected_items[label] = detected_items.get(label, 0) + 1

    # 4️⃣ 儲存到資料庫
    DB_PATH = os.path.join(os.path.dirname(__file__), 'detections.db')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    time_now_full = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for label, count in detected_items.items():
        cursor.execute('''
            INSERT INTO detections (image_name, timestamp, label, count)
            VALUES (?, ?, ?, ?)
        ''', (image_name, time_now_full, label, count))

    conn.commit()
    conn.close()

    # 5️⃣ 在圖片上畫框
    for *box, conf, cls in pred.tolist():
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        color = (0, 255, 0)
        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, lineType=cv2.LINE_AA)

    result_img_path = os.path.join(UPLOAD_FOLDER, f"result_{image_name}")
    Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)).save(result_img_path)

    # 6️⃣ 記憶體釋放
    del pred
    del img_tensor
    del model
    torch.cuda.empty_cache()

    # 7️⃣ 回傳 LINE 訊息
    time_now = datetime.datetime.now().strftime('%H:%M')
    message_text = f"辨識時間：{time_now}\n"
    for label, count in detected_items.items():
        message_text += f"{label}: {count}\n"

    BASE_URL = os.getenv("BASE_URL", "https://yolo-line-render.onrender.com")
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
