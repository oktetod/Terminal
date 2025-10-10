# Filename: Dockerfile

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Tambahkan 'asyncpg' untuk koneksi PostgreSQL
RUN pip install --no-cache-dir \
    "cryptg>=0.4.0" \
    "cerebras-cloud-sdk" \
    "aiohttp" \
    "asyncpg" \
    "telethon>=1.36.0" \
    "python-dotenv>=1.0.0"

# Copy kedua file aplikasi
COPY database.py .
COPY bot.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "bot.py"]
