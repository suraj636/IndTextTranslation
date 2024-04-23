from fastapi import FastAPI, Form, HTTPException
from transformers import MarianMTModel, MarianTokenizer
import logging

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

languages = ["hi", "ar", "ur", "tl"]

def load_model(language):
    model = MarianMTModel.from_pretrained(f"./Indian/{language}")
    tokenizer = MarianTokenizer.from_pretrained(f"./Indian/{language}")
    return model, tokenizer

def translate_text(text, language):
    if language not in languages:
        raise HTTPException(status_code=400, detail=f"Language '{language}' not supported")

    model, tokenizer = load_model(language)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_ids = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
    return translated_text[0]

@app.get("/")
async def root():
    return {"message": "Welcome to the translation API for Indian Languages"}

@app.post("/translate/")
async def translate_text_api(text: str = Form(...), language: str = Form(...)):
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    if len(text) > 512:
        raise HTTPException(status_code=400, detail="Text input is too long, maximum length is 512 characters")
    try:
        translated_text = translate_text(text, language)
        return {"translated_text": translated_text}
    except Exception as e:
        logging.exception("Translation failed")
        raise HTTPException(status_code=500, detail="Translation failed")
