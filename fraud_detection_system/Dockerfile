FROM python:3.11-slim

WORKDIR /app

# libgomp потрібна XGBoost та scikit-learn для багатопотоковості
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app

EXPOSE 8000 8501

# За замовчуванням запускається API; дашборд і тренер задаються в docker-compose.yml
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
