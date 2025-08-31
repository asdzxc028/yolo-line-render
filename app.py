from flask import Flask, request, Response
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage,ImageSendMessage
import requests, traceback
from io import BytesIO
import os
from datetime import datetime

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
HF_API_URL = f"https://{HF_SPACE_NAME}.hf.space/api/detect"
HF_DB_URL = f"https://{HF_SPACE_NAME}.hf.space/api/download_db"

# LINE Webhook 路由
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    print("📩 收到 Webhook 請求：", body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("❌ Webhook 驗證失敗")
        return 'Invalid signature', 403
    except Exception as e:
        print(f"❌ Callback 處理失敗：{e}")
        return 'Internal Server Error', 500
    return 'OK'

# 處理圖片訊息的方式
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    try:
        # 下載 LINE 傳來的圖片
        message_content = line_bot_api.get_message_content(event.message.id)
        image_bytes = BytesIO(message_content.content)

        # 使用圖片 ID + 時間命名
        message_id = event.message.id
        timestamp = datetime.now().strftime("%H%M")
        filename = f"{message_id}_{timestamp}.jpg"
        
        # 上傳給 Hugging Face API
        files = {
            "file": (filename, image_bytes, "image/jpeg")
        }
        headers = {"User-Agent": "LineYOLOBot/1.0"}
        res = requests.post(HF_API_URL, files=files, headers=headers, timeout=20)
        res.raise_for_status()

        result = res.json()
        
        # 取得文字與圖片 URL
        message_text = result.get("message", "⚠️ 沒有回傳 message")
        image_url = result.get("image_url", "/file/default.jpg")
        thumb_url = result.get("thumb_url", image_url)

        # 若是完整 URL，直接使用；否則補上完整域名
        if not image_url.startswith("http"):
            full_image_url = f"https://{HF_SPACE_NAME}.hf.space{image_url}"
        else:
            full_image_url = image_url

        if not thumb_url.startswith("http"):
            full_thumb_url = f"https://{HF_SPACE_NAME}.hf.space{thumb_url}"
        else:
            full_thumb_url = thumb_url
        # 回覆給觸發的人，表示 Bot 已收到訊息
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="📸 圖片處理中，請稍候...")
        )

        # 圖片處理後，主動推播給整個群組或個人
        smart_push_message(event, [
            TextSendMessage(text=message_text),
            ImageSendMessage(
                original_content_url=full_image_url,
                preview_image_url=full_thumb_url  
            ),
            TextSendMessage(text=f"📥 下載完整資料庫：{HF_DB_URL}")
        ])

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
    
def smart_push_message(event, messages):
    source = event.source
    try:
        if source.type == "group" and source.group_id:
            line_bot_api.push_message(source.group_id, messages)
        elif source.type == "user" and source.user_id:
            line_bot_api.push_message(source.user_id, messages)
        else:
            print("⚠️ 無法推送訊息，未知來源：", source)
    except Exception as e:
        print(f"❌ 推送訊息失敗：{e}")

@app.route("/", methods=["GET"])
def index():
    return "🚀 LINE YOLO Bot 正在運行中", 200

# 啟動程式
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

