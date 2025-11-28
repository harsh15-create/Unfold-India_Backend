from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import Response
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY_CHAT = os.getenv("GROQ_API_KEY")
API_KEY_TRANSLATE = os.getenv("GROQ_API_KEY")

API_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ CHATBOT ------------------
SYSTEM_PROMPT = """
You are an AI travel buddy specializing in budget-friendly journeys across INDIA ONLY behave like an expert indian travel guide.
NEVER suggest foreign destinations unless the user asks specifically.
If the user requests an itinerary:
- For 3-4 day itineraries (traveler mode), provide a COMPLETE detailed plan with:
  Day-wise breakdown, best time to visit each place, local food recommendations like some old food shops and trendy cafes, transport methods, estimated budget, stay options, safety tips, and local insights and best hidden gems.
- For 1-day itineraries (normal mode), provide a quick sightseeing route focusing on best attractions.
- Make responses ENGAGING and exciting, but still practical and helpful.
- Always keep responses structured and easy to read.
"""

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    msg = data.get("message")
    if not msg:
        return JSONResponse(status_code=400, content={"error": "Message field required"})

    # detect itinerary request
    is_itinerary = "itinerary" in msg.lower()
    payload = {
        "model": "llama-3.1-8b-instant",
        "max_tokens": 2000 if is_itinerary else 2000,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg}
        ]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY_CHAT}",
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(API_URL, json=payload, headers=headers).json()
        reply = res["choices"][0]["message"]["content"]
        return {"reply": reply}
    except Exception as e:
        print("CHAT ERROR:", e)
        return {"reply": "Chatbot error."}


# ------------------ TRANSLATION ------------------

LANG_MAP = {
    "hi": "Hindi",
    "ta": "Tamil",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati"
}

@app.post("/api/translate")
async def translate(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    text = data.get("text")
    target_code = data.get("target")

    target_lang = LANG_MAP.get(target_code, "Hindi")

    # STRICT JSON response prompt
    system_prompt = (
        f"You are a translation engine. Translate the user's sentence into {target_lang}. "
        f"ALWAYS respond in valid JSON ONLY in this format:\n"
        f'{{"translated": "TRANSLATED_TEXT"}}\n'
        f"No explanations. No extra text. No commentary."
    )

    payload = {
        "model": "llama-3.1-8b-instant",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY_TRANSLATE}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers).json()

        raw = response["choices"][0]["message"]["content"]

        # LLaMA returns JSON NOW, so we parse it safely
        translated = json.loads(raw)["translated"]

        return {"translated": translated}

    except Exception as e:
        print("TRANS ERROR:", e)
        print("RAW:", response)
        return {"translated": "Translation error."}

@app.post("/api/tts")
async def tts(request: Request):
    try:
        data = await request.json()
        text = data.get("text")
        voice_id = data.get("voiceId", "en-US-ryan")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    tts_payload = {
        "model": "tts-1",
        "voice": voice_id,
        "input": text
    }

    headers = {
        "Authorization": f"Bearer {API_KEY_CHAT}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    try:
        resp = requests.post("https://api.groq.com/openai/v1/audio/speech", json=tts_payload, headers=headers)
        print("STATUS:", resp.status_code)
        print("HEADERS:", resp.headers)
        print("LENGTH:", len(resp.content))
        return Response(content=resp.content, media_type="audio/mpeg")
    except Exception as e:
        print("TTS ERROR:", e)
        return JSONResponse(status_code=500, content={"error": "TTS error"})
