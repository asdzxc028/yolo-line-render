from flask import Flask, request, Response
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, ImageSendMessage
import requests, traceback
from io import BytesIO
from PIL import Image
import os
import base64

app = Flask(__name__)

# 環境變數
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# 驗證
if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError("❌ 請設定 LINE_CHANNEL_SECRET 和 LINE_CHANNEL_ACCESS_TOKEN 環境變數")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Hugging Face 設定
HF_SPACE_NAME = "ylrasd-yolo-line-render"
HF_API_URL = f"https://{HF_SPACE_NAME}.hf.space/api/predict/detect"
HF_DB_URL = f"https://{HF_SPACE_NAME}.hf.space/api/download_db"

# LINE Webhook 路由
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return 'Invalid signature', 403
    return 'OK'

# 處理圖片訊息的方式
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    try:
        message_content = line_bot_api.get_message_content(event.message.id)
        image = Image.open(BytesIO(message_content.content))

        # 圖片轉 base64 傳給 Hugging Face Space
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_str = f"data:image/jpeg;base64,{img_base64}"
        
        # 發出 POST 請求
        payload = {"data": [img_str]}
        headers = {"User-Agent": "LineYOLOBot/1.0"}
        res = requests.post(HF_API_URL, json=payload, headers=headers, timeout=20)

        if res.status_code != 200:
            print(f"❌ API 錯誤內容: {res.text}")
            message_text = f"⚠️ YOLO 服務錯誤：{res.status_code}"
            image_url = "https://placekitten.com/300/300"
        else:
            try:
                result = res.json()
            except ValueError:
                result = {}
            message_text = result.get("data", [{}])[0].get("message", "⚠️ 沒有回傳 message")
            image_url = result.get("data", [{}])[0].get("image_url", "https://placekitten.com/300/300")

        # 回傳到 LINE
        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=message_text),
                ImageSendMessage(original_content_url=image_url, preview_image_url=image_url),
                TextSendMessage(text="📥 下載完整資料庫：https://yolo-line-render.onrender.com/download_db")
            ]
        )

    except Exception as e:
        print(f"🔥 發生例外：{e}")
        print(traceback.format_exc())
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="❌ 發生錯誤，請稍後再試")
        )

# 資料庫下載
@app.route("/download_db", methods=["GET"])
def download_db():
    try:
        r = requests.get(HF_DB_URL, stream=True, timeout=10)
        if r.status_code == 200:
            return Response(
                r.iter_content(chunk_size=8192),
                content_type=r.headers.get("content-type", "application/octet-stream"),
                headers={"Content-Disposition": "attachment; filename=detections.db"}
            )
        else:
            return "❌ 從 Hugging Face Space 抓不到資料庫", 404
    except requests.exceptions.RequestException as e:
        print(f"❌ 抓取資料庫時發生錯誤：{e}")
        return "❌ 資料庫服務目前無法連線，請稍後再試", 500
    
@app.route("/", methods=["GET"])
def index():
    return "🚀 LINE YOLO Bot 正在運行中", 200

# 啟動程式
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

