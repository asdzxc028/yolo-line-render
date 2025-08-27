from flask import Flask, request, abort, Response
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, ImageSendMessage
import requests, base64, traceback
from io import BytesIO
from PIL import Image

app = Flask(__name__)

LINE_CHANNEL_SECRET = "LINE_CHANNEL_SECRET"
LINE_CHANNEL_ACCESS_TOKEN = "LINE_CHANNEL_ACCESS_TOKEN"

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

HF_SPACE_NAME = "ylrasd-yolo-line-render"
HF_API_URL = f"https://{HF_SPACE_NAME}.hf.space/run/predict"
HF_DB_URL = f"https://{HF_SPACE_NAME}.hf.space/file/static/uploads/detections.db"

# 🔥 全域 Exception 捕捉，方便 debug
@app.errorhandler(Exception)
def handle_exception(e):
    print("🔥 Exception:", e)
    traceback.print_exc()
    return "Internal Server Error", 500

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image = Image.open(BytesIO(message_content.content))

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # 呼叫 Hugging Face Space
    response = requests.post(HF_API_URL, json={"data": [img_str]})
    print("📡 HF 回傳內容:", response.text)  # 👈 先印出來確認格式
    result = response.json()["data"][0]

    message_text = result["message"]
    image_url = result["image_url"]

    line_bot_api.reply_message(
        event.reply_token,
        [
            TextSendMessage(text=message_text),
            ImageSendMessage(original_content_url=image_url, preview_image_url=image_url),
            TextSendMessage(text="📥 下載完整資料庫：\https://yolo-line-render.onrender.com/download_db")
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

