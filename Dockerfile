# ---- base image ----
FROM python:3.11-slim

# ---- install deps ----
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- copy app & model ----
COPY api/ /app/api/
COPY distilbert-imdb/ /app/distilbert-imdb/

# ---- expose + run ----
WORKDIR /app
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
