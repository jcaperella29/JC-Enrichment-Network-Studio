FROM python:3.11-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    BROWSER_PATH=/usr/bin/chromium \
    CHROME_PATH=/usr/bin/chromium

# Chromium is required by Kaleido v1+ for Plotly SVG/PDF export.
# fonts-liberation helps exported figures render text cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    fonts-liberation \
    ca-certificates \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY TEST.csv .

EXPOSE 8080

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-8080} app:server --workers 2 --threads 4 --timeout 180"]
