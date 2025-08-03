# IMDb Sentiment (DistilBERT)

**Task** Binary movie-review sentiment  
**Model** `distilbert-base-uncased` fine-tuned for 3 epochs

| Split       | Accuracy |
|-------------|----------|
| Validation† | 0.926 |
| Test‡       | 0.981 |

† 5 000 reviews (10 % split) • ‡ 25 000 reviews (official IMDb test set)

---

## Quick start

```bash
# Docker
docker build -t imdb-sentiment .
docker run -p 8000:80 imdb-sentiment
# → http://localhost:8000/docs  for Swagger UI

# Local Python
python -m venv .venv && .\.venv\Scripts\activate   # on Windows
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Example request

```http
POST /predict
{ "text": "Surprisingly good movie!" }
→ { "label": "positive", "confidence": 0.97 }
```

## Model weights (>100 MB)

The weight file is kept outside the Git repo.  
Download it once and place it under `distilbert-imdb/`:

```bash
curl -L -o distilbert-imdb/model.safetensors \
  https://github.com/<USER>/imdb-sentiment/releases/download/v1.0.0/model.safetensors
```

(Or let the app fetch it automatically on first run.)

## Project structure

```text
imdb-sentiment/
├ api/               # FastAPI app
├ distilbert-imdb/   # fine-tuned weights & tokenizer
├ Dockerfile
├ requirements.txt
└ README.md
```
