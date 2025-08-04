# IMDb Sentiment (DistilBERT)

**Task** Binary sentiment on IMDb movie reviews  
**Model** `distilbert-base-uncased`, fine-tuned **3 epochs**  
&nbsp; • cosine LR decay • 5 % warm-up • max len 384 • effective batch 32

| Split | Size | Accuracy |
|-------|------|----------|
| Validation† | 2 500 | **0.920** |
| Test‡ | 25 000 | **0.919** |

† 10 % of the 25 k official *train* split (held out during training)  
‡ Canonical 25 k IMDb *test* split — never seen while tuning

---

## Quick start

```bash
# ── Docker ─────────────────────────────────────────────
docker build -t imdb-sentiment .
docker run -p 8000:80 imdb-sentiment          # → http://localhost:8000/docs

# ── Local Python (Linux/macOS) ─────────────────────────
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload

# Windows PowerShell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api.main:app --reload
````

## Example request

```http
POST /predict
{ "text": "Surprisingly good movie!" }

→ 200 OK
  { "label": "positive", "confidence": 0.97 }
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
├ api/                 FastAPI service (POST /predict)
├ distilbert-imdb/     ← place model.safetensors + tokenizer files here
├ Dockerfile           Container build
├ notebooks/           Training & evaluation recipe
├ requirements.txt
└ README.md
```



