import os
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

class ChatRequest(BaseModel):
    message: str
    history: List[dict]
    system_prompt: str
    config: dict

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not model and not req.config.get("apiKey"):
        raise HTTPException(status_code=400, detail="API Key not configured")
    
    # Use user key if provided, otherwise system key
    active_model = model
    if req.config.get("apiKey"):
        genai.configure(api_key=req.config.get("apiKey"))
        active_model = genai.GenerativeModel('gemini-1.5-flash')

    try:
        # Prepare history in Gemini format
        chat = active_model.start_chat(history=[])
        
        # Inject system prompt as initial instructions if possible or as part of first msg
        prompt = f"{req.system_prompt}\n\nUser Command: {req.message}"
        
        response = await asyncio.to_thread(chat.send_message, prompt)
        
        return {
            "text": response.text,
            "status": "success"
        }
    except Exception as e:
        print(f"AI Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "online", "core": "python/fastapi"}

# Serve static files in production
if os.path.exists("dist"):
    app.mount("/", StaticFiles(directory="dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Port 3000 is mandatory for the environment proxy
    uvicorn.run(app, host="0.0.0.0", port=3000)
