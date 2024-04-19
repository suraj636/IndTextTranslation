# Useful for handling form data and HTTP Exceptions
from fastapi import FastAPI, Form, HTTPException
# These are used for handling cross-origin resource sharing
from fastapi.middleware.cors import CORSMiddleware
# Importing libraries from HuggingFaceTransformer for machine translational task
from transformers import MarianMTModel, MarianTokenizer
# Imports the logging module, which provides a flexible
# framework for emitting log messages from Python programs.
import logging

# Created a FastAPI application instance named app
app = FastAPI()

# Configures the logging system to output log messages with
# the DEBUG level or higher.
logging.basicConfig(level=logging.DEBUG)

# Allowing CORS for development purposes
# Adds CORS middleware to the FastAPI application.
# This middleware allows cross-origin requests from any origin ("*"),
# including credentials (cookies, authorization headers), for POST requests,
# and allows any headers to be sent
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load fine-tuned models and tokenizers for each language
models = {
    "hi": MarianMTModel.from_pretrained("./Indian/hi"),
    "ar":MarianMTModel.from_pretrained("./Indian/ar"),
    "ur":MarianMTModel.from_pretrained("./Indian/ur"),
    "tl":MarianMTModel.from_pretrained("./Indian/tl")
}

tokenizers = {
    "hi": MarianTokenizer.from_pretrained("./Indian/hi"),
    "ur": MarianTokenizer.from_pretrained("./Indian/ur"),
    "ar": MarianTokenizer.from_pretrained("./Indian/ar"),
    "tl": MarianTokenizer.from_pretrained("./Indian/tl")
}

# Default route
@app.get("/")
async def root():
    return {"message": "Welcome to the translation API for Indian Languages"}

# Define translation function
def translate_text(text, language):
    model = models.get(language)
    tokenizer = tokenizers.get(language)
    if model is None or tokenizer is None:
        raise HTTPException(status_code=400, detail=f"Language '{language}' not supported")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_ids = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
    return translated_text[0]

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