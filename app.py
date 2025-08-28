from flask import Flask, request, Response
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, ImageSendMessage
import requests, traceback
from io import BytesIO
from PIL import Image
import os
app = Flask(__name__)

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError("❌ 請設定 LINE_CHANNEL_SECRET 和 LINE_CHANNEL_ACCESS_TOKEN 環境變數")
HF_SPACE_NAME = "ylrasd-yolo-line-render"
HF_API_URL = f"https://{HF_SPACE_NAME}.hf.space/run/detect"
HF_DB_URL = f"https://{HF_SPACE_NAME}.hf.space/static/uploads/detections.db"

# 🔥 全域 Exception 捕捉，方便 debug
@app.errorhandler(Exception)
def handle_exception(e):
    print("🔥 Exception:", e)
    traceback.print_exc()
    return "Internal Server Error", 500

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("❌ 簽章驗證失敗")
        return "Invalid signature", 400
    except Exception as e:
        print("🔥 Callback exception:", e)
        traceback.print_exc()
        # ❗ 總是回 200，避免 LINE webhook 停掉
        return "OK", 200

    return "OK", 200
@app.route("/", methods=["GET"])
def index():
    return "✅ LINE Bot Server is running", 200

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image = Image.open(BytesIO(message_content.content))

    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    # 呼叫 Hugging Face Space
    files = {"image": ("input.jpg", buffered.getvalue(), "image/jpeg")}
    response = requests.post(HF_API_URL, files=files)

    print("📡 HF 回傳內容:", response.text)

    if response.status_code != 200:
        message_text = f"⚠️ YOLO 服務錯誤：{response.status_code}"
        image_url = "https://placekitten.com/300/300"
    else:
        try:
            result = response.json()
            message_text = result.get("message", "⚠️ YOLO 沒有回傳 message")
            image_url = result.get("image_url", "https://placekitten.com/300/300")
        except Exception as e:
            print("🔥 JSON 解析錯誤:", e)
            message_text = "⚠️ YOLO 回傳資料異常"
            image_url = "https://placekitten.com/300/300"
        # 🔹 回覆 LINE 使用者
    line_bot_api.reply_message(
        event.reply_token,
        [
            TextSendMessage(text=message_text),
            ImageSendMessage(original_content_url=image_url, preview_image_url=image_url),
            TextSendMessage(text="📥 下載完整資料庫：https://yolo-line-render.onrender.com/download_db")
        ]
    )

@app.route("/download_db", methods=["GET"])
def download_db():
    r = requests.get(HF_DB_URL, stream=True)
    if r.status_code == 200:
        return Response(
            r.iter_content(chunk_size=8192),
            content_type=r.headers.get("content-type", "application/octet-stream"),
            headers={"Content-Disposition": "attachment; filename=detections.db"}
        )
    else:
        return "❌ 從 Hugging Face Space 抓不到資料庫", 404

if __name__ == "__main__":
    app.run(port=5000)

