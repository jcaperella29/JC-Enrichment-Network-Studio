FROM python:3.11-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (safe baseline for scientific wheels if any fallback builds happen)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:server", "--workers", "2", "--threads", "4", "--timeout", "120"]
