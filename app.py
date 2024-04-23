from fastapi import FastAPI, Form, HTTPException
from transformers import MarianMTModel, MarianTokenizer
import logging

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

models = {}
tokenizers = {}

languages = ["hi", "ar", "ur", "tl"]

def load_models():
    for lang in languages:
        models[lang] = MarianMTModel.from_pretrained(f"./Indian/{lang}")
        tokenizers[lang] = MarianTokenizer.from_pretrained(f"./Indian/{lang}")

def lazy_load_models():
    if not models:
        load_models()

def translate_text(text, language):
    lazy_load_models()
    model = models.get(language)
    tokenizer = tokenizers.get(language)
    if not model or not tokenizer:
        raise HTTPException(status_code=400, detail=f"Language '{language}' not supported")

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
