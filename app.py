from flask import Flask, request, abort, send_from_directory
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, ImageSendMessage
from linebot.exceptions import InvalidSignatureError

import datetime
import requests
import os
from PIL import Image
import uuid

import cv2
import numpy as np

# 如果你要使用 YOLO 模型辨識
from ultralytics import YOLO

app = Flask(__name__)

# LINE API 金鑰
LINE_CHANNEL_ACCESS_TOKEN = 'WA1bLzN38CJbpKWEG9+ZNNzQQEZ0CR6zyiQGynhky2HVQem/LkI1HZgAfSsKImhIQ1CsBEKkicr+R7Sd5Pu1EvpXGa1xRN0/xmC+ePhMsMP2rxXFXQWhIbdn8uAtAXHD8pjDeIsfUJ4lG2gudvyc/gdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = '36e4a3ed3f33de2ecf6f703bdecdf4ae'
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 載入 YOLO 模型
model = YOLO('animals.pt')  # 可以換成你訓練好的模型

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id

    # 1️⃣ 下載圖片
    image_content = line_bot_api.get_message_content(message_id)
    image_name = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, image_name)

    with open(image_path, 'wb') as f:
        for chunk in image_content.iter_content():
            f.write(chunk)

    # 2️⃣ 用 YOLO 分析圖片
    result = model(image_path)[0]
    boxes = result.boxes
    labels = model.names

    # 3️⃣ 統計辨識結果
    detected_items = {}
    for box in boxes:
        cls = int(box.cls[0])
        label = labels[cls]
        detected_items[label] = detected_items.get(label, 0) + 1

    # 4️⃣ 用 OpenCV 自行畫框（不顯示信心值）
    result_path = os.path.join(UPLOAD_FOLDER, f"result_{image_name}")
    img = cv2.imread(image_path)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        label = labels[cls]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imwrite(result_path, img)

    # 5️⃣ 組合回傳訊息
    time_now = datetime.datetime.now().strftime('%H:%M')
    message_text = f"辨識時間：{time_now}\n"
    for label, count in detected_items.items():
        message_text += f"{label}: {count}\n"

    # 6️⃣ 準備回傳
    image_url = f"https://你的render專案.onrender.com/static/uploads/result_{image_name}"

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
    app.run()
