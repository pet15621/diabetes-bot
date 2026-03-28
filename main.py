from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage
)
import pandas as pd
import numpy as np
import joblib
import shap
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ========== โหลด Line credentials ==========
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler      = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# ========== โหลด Models + Scaler ==========
models = {
    "1": ("Logistic Regression", joblib.load("models/logistic.pkl")),
    "2": ("Decision Tree",       joblib.load("models/decision_tree.pkl")),
    "3": ("ANN (MLP)",           joblib.load("models/ann.pkl")),
}
scaler       = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")
background   = joblib.load("models/shap_background.pkl")

# ========== คำถามทีละขั้น ==========
QUESTIONS = [
    ("Pregnancies",              "จำนวนครั้งที่ตั้งครรภ์ (ถ้าไม่เคยใส่ 0)"),
    ("Glucose",                  "ระดับน้ำตาลในเลือด (mg/dL) เช่น 120"),
    ("BloodPressure",            "ความดันโลหิต Diastolic (mmHg) เช่น 72"),
    ("SkinThickness",            "ความหนาผิวหนังบริเวณต้นแขน (mm) เช่น 23"),
    ("Insulin",                  "ระดับอินซูลิน (µU/ml) เช่น 80"),
    ("BMI",                      "ค่า BMI เช่น 28.5"),
    ("DiabetesPedigreeFunction", "ความเสี่ยงทางพันธุกรรม (0.0–2.5) เช่น 0.5"),
    ("Age",                      "อายุ (ปี) เช่น 35"),
]

# ========== เก็บ session ของแต่ละ user ==========
sessions = {}

# ========== SHAP helper ==========
def get_shap_top3(model_key, input_scaled, input_raw):
    name, model = models[model_key]
    X_df = pd.DataFrame([input_scaled], columns=feature_names)

    if model_key == "1":
        explainer  = joblib.load("models/explainer_lr.pkl")
        sv         = explainer.shap_values(X_df)
        sv1        = np.array(sv).flatten()
    elif model_key == "2":
        explainer  = joblib.load("models/explainer_dt.pkl")
        sv         = explainer.shap_values(X_df)
        sv1        = np.array(sv[1] if isinstance(sv, list) else sv).flatten()
    else:
        explainer  = shap.KernelExplainer(model.predict_proba, background)
        sv         = explainer.shap_values(X_df, nsamples=100)
        sv1        = np.array(sv).flatten()[8:]

    impact = pd.Series(np.abs(sv1), index=feature_names)
    top3   = impact.nlargest(3)

    lines = []
    for feat, _ in top3.items():
        val = input_raw[feat]
        lines.append(f"  • {feat}: {val:.1f}")
    return lines

# ========== predict helper ==========
def predict(model_key, answers):
    name, model = models[model_key]
    input_raw    = {q[0]: float(answers[i]) for i, q in enumerate(QUESTIONS)}
    input_scaled = scaler.transform([list(input_raw.values())])[0]
    prob         = model.predict_proba([input_scaled])[0][1]
    top3_lines   = get_shap_top3(model_key, input_scaled, input_raw)

    if prob >= 0.7:
        risk = "ความเสี่ยงสูง"
    elif prob >= 0.4:
        risk = "ความเสี่ยงปานกลาง"
    else:
        risk = "ความเสี่ยงต่ำ"

    top3_text = "\n".join(top3_lines)
    msg = (
        f"ผลจากคุณหมอ {name}\n"
        f"{'─' * 28}\n"
        f"ความเสี่ยงเบาหวาน: {prob*100:.1f}%\n"
        f"ระดับ: {risk}\n\n"
        f"3 ปัจจัยหลักที่ส่งผล:\n{top3_text}\n\n"
        f"หมายเหตุ: ผลนี้เป็นการประเมินเบื้องต้น\n"
        f"ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่แม่นยำ"
    )
    return msg

# ========== Webhook endpoint ==========
@app.get("/webhook")
async def webhook_verify():
    return JSONResponse(content={"status": "ok"})

@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body      = await request.body()
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        return JSONResponse(status_code=400,
                            content={"message": "Invalid signature"})
    return JSONResponse(content={"message": "OK"})

# ========== Message handler ==========
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    text    = event.message.text.strip()

    # ========== เริ่มต้น / เลือกโมเดล ==========
    if text in ["เริ่ม", "start", "เริ่มต้น", "สวัสดี", "hi", "hello"]:
        sessions[user_id] = {"step": "choose_model", "answers": []}
        reply = (
            "สวัสดีครับ! ยินดีต้อนรับสู่ระบบประเมินความเสี่ยงเบาหวาน\n\n"
            "กรุณาเลือกโมเดลที่ต้องการ:\n\n"
            "1 - Logistic Regression\n"
            "    (เร็ว อธิบายง่าย)\n\n"
            "2 - Decision Tree\n"
            "    (เห็นเหตุผลชัดเจน)\n\n"
            "3 - ANN (MLP)\n"
            "    (แม่นยำสูงสุด)\n\n"
            "พิมพ์ 1, 2 หรือ 3 เพื่อเลือก"
        )
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=reply))
        return

    # ========== ถ้ายังไม่มี session ==========
    if user_id not in sessions:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='พิมพ์ "เริ่ม" เพื่อเริ่มประเมินความเสี่ยงเบาหวานครับ'))
        return

    session = sessions[user_id]

    # ========== เลือกโมเดล ==========
    if session["step"] == "choose_model":
        if text not in ["1", "2", "3"]:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="กรุณาพิมพ์ 1, 2 หรือ 3 เท่านั้นครับ"))
            return

        session["model_key"] = text
        session["step"]      = 0
        model_name           = models[text][0]
        q_text               = QUESTIONS[0][1]

        reply = (
            f"เลือก {model_name} แล้วครับ\n\n"
            f"คำถามที่ 1/8\n"
            f"{q_text}"
        )
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=reply))
        return

    # ========== รับคำตอบทีละข้อ ==========
    if isinstance(session["step"], int):
        step = session["step"]

        # ตรวจว่าเป็นตัวเลขไหม
        try:
            float(text)
        except ValueError:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=f"กรุณากรอกเป็นตัวเลขเท่านั้นครับ\n\n"
                         f"คำถามที่ {step+1}/8\n"
                         f"{QUESTIONS[step][1]}"))
            return

        session["answers"].append(text)
        next_step = step + 1

        # ยังไม่ครบ 8 ข้อ
        if next_step < len(QUESTIONS):
            session["step"] = next_step
            reply = (
                f"คำถามที่ {next_step+1}/8\n"
                f"{QUESTIONS[next_step][1]}"
            )
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text=reply))

        # ครบ 8 ข้อ → predict
        else:
            try:
                result = predict(session["model_key"], session["answers"])
                line_bot_api.reply_message(
                    event.reply_token, TextSendMessage(text=result))
            except Exception as e:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=f"เกิดข้อผิดพลาดครับ: {str(e)}"))

            # ถามว่าจะประเมินใหม่ไหม
            del sessions[user_id]
            line_bot_api.push_message(
                user_id,
                TextSendMessage(
                    text='พิมพ์ "เริ่ม" เพื่อประเมินใหม่อีกครั้งได้เลยครับ'))

# ========== Health check ==========
@app.get("/")
def root():
    return {"status": "running", "message": "Diabetes Bot API is ready"}