"""
Render deploy note:
Set Start Command to:
python -m uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load secrets
load_dotenv()

# THE FIX: Force the library to use a stable endpoint
os.environ["GOOGLE_API_USE_MTLS"] = "never"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use a model with free tier quota (gemini-2.0-flash has limit: 0 on free tier)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoteRequest(BaseModel):
    text: str

@app.post("/generate")
async def generate_cards(request: NoteRequest):
    prompt = f"""
    Create 5 multiple choice questions (MCQs) from the text below.
    Each MCQ must have exactly 4 options, with one correct answer.
    Return ONLY valid JSON in this exact format (no markdown, no extra text):
    [{{"question": "Question text?", "options": ["Option A", "Option B", "Option C", "Option D"], "correctAnswer": "Option A"}}]
    The correctAnswer must be exactly one of the option strings.
    
    TEXT: {request.text}
    """
    
    try:
        # We add a safety check here
        response = model.generate_content(prompt)
        if not response.text:
            raise Exception("AI returned an empty response")
            
        json_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(json_text)
    except Exception as e:
        print(f"DEBUG ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)