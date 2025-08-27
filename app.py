from flask import Flask, request, abort, Response
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, ImageSendMessage
import requests, base64
from io import BytesIO
from PIL import Image
import os
app = Flask(__name__)

LINE_CHANNEL_SECRET =os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN =os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ⚡ 換成你的 Hugging Face Space 名字
HF_SPACE_NAME = "ylrasd-yolo-line-render"
HF_API_URL = f"https://{HF_SPACE_NAME}.hf.space/run/predict"

# Hugging Face Space 上的 DB URL
HF_DB_URL = f"https://{HF_SPACE_NAME}.hf.space/file/static/uploads/detections.db"

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
    # 下載使用者傳的圖片
    message_content = line_bot_api.get_message_content(event.message.id)
    image = Image.open(BytesIO(message_content.content))

    # 轉成 base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # 呼叫 HF Space YOLO API
    response = requests.post(HF_API_URL, json={"data": [img_str]})
    result = response.json()["data"][0]

    # 取出辨識文字與圖片網址
    message_text = result["message"]
    image_url = result["image_url"]

    # 回傳 LINE 訊息（文字 + 圖片 + DB 下載連結）
    line_bot_api.reply_message(
        event.reply_token,
        [
            TextSendMessage(text=message_text),
            ImageSendMessage(original_content_url=image_url, preview_image_url=image_url),
            TextSendMessage(text="📥 下載完整資料庫：\nhttps://你的伺服器/download_db")
        ]
    )

# ➤ Flask 串流轉發 Hugging Face Space 的 detections.db
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
