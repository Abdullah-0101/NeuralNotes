"""
Render deploy note:
Set Start Command to:
python -m uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import json
import base64
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

os.environ["GOOGLE_API_USE_MTLS"] = "never"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
    language: str = "English"
    image: Optional[str] = None
    mode: str = "mcq"

class MindMapRequest(BaseModel):
    text: str
    language: str = "English"


def build_prompt(text: str, lang: str, mode: str, has_image: bool = False) -> str:
    lang_instruction = (
        "Generate the question, all options, and the correctAnswer in clear, modern Arabic."
        if lang == "Arabic"
        else "Generate everything in English."
    )

    ocr_prefix = (
        "First, perform OCR on the attached image to extract all readable text. "
        "Then use that extracted text (combined with any additional notes below) to "
        if has_image else ""
    )

    count = "5 to 10" if has_image else "5"

    if mode == "study":
        return f"""
{ocr_prefix}Create {count} study flashcards.
Each flashcard must have a question and a detailed answer.
{lang_instruction}
Return ONLY valid JSON in this exact format (no markdown, no extra text):
[{{"question": "...", "answer": "..."}}]

TEXT: {text}
"""
    return f"""
{ocr_prefix}Create {count} multiple choice questions (MCQs).
Each MCQ must have exactly 4 options, with one correct answer.
{lang_instruction}
Return ONLY valid JSON in this exact format (no markdown, no extra text):
[{{"question": "Question text?", "options": ["Option A", "Option B", "Option C", "Option D"], "correctAnswer": "Option A"}}]
The correctAnswer must be exactly one of the option strings.

TEXT: {text}
"""


@app.post("/generate")
async def generate_cards(request: NoteRequest):
    lang = request.language if request.language in ("English", "Arabic") else "English"
    mode = request.mode if request.mode in ("mcq", "study") else "mcq"
    has_image = bool(request.image)

    prompt = build_prompt(request.text, lang, mode, has_image)

    try:
        content_parts = [prompt]

        if request.image:
            img_data = request.image
            mime_type = "image/png"
            if "," in img_data:
                header, img_data = img_data.split(",", 1)
                if "jpeg" in header or "jpg" in header:
                    mime_type = "image/jpeg"
                elif "webp" in header:
                    mime_type = "image/webp"
            img_bytes = base64.b64decode(img_data)
            content_parts.append({
                "mime_type": mime_type,
                "data": img_bytes
            })

        response = model.generate_content(content_parts)
        if not response.text:
            raise Exception("AI returned an empty response")

        json_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(json_text)
    except Exception as e:
        print(f"DEBUG ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-mindmap")
async def generate_mindmap(request: MindMapRequest):
    lang = request.language if request.language in ("English", "Arabic") else "English"
    lang_instruction = (
        "Generate all labels in clear, modern Arabic."
        if lang == "Arabic"
        else "Generate all labels in English."
    )

    prompt = f"""
Analyze the following text and create a hierarchical mind map.
{lang_instruction}
Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "label": "Main Topic",
  "children": [
    {{
      "label": "Sub-topic 1",
      "children": [
        {{"label": "Key Point A", "children": []}},
        {{"label": "Key Point B", "children": []}}
      ]
    }},
    {{
      "label": "Sub-topic 2",
      "children": [
        {{"label": "Key Point C", "children": []}}
      ]
    }}
  ]
}}
Each node must have "label" (string) and "children" (array).
Create 3-6 sub-topics, each with 2-4 key points.

TEXT: {request.text}
"""

    try:
        response = model.generate_content(prompt)
        if not response.text:
            raise Exception("AI returned an empty response")
        json_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(json_text)
    except Exception as e:
        print(f"DEBUG ERROR (mindmap): {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)