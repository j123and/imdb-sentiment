from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


MODEL_PATH = "distilbert-imdb"          # <-- just this

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval().to("cpu")                          # GPU not needed for demo

label_map = {0: "negative", 1: "positive"}

# --- FastAPI app ----------------------------------------------------------
app = FastAPI(title="IMDb Sentiment API")

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    inputs = tokenizer(
        item.text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        pred   = int(torch.argmax(logits, dim=-1))
        score  = float(torch.softmax(logits, dim=-1)[0, pred])
    return {"label": label_map[pred], "confidence": round(score, 4)}
