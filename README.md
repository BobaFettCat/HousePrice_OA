# California Housing Price Prediction Service

This directory contains the production-ready machine learning estimation microservice.

## 1. Chosen Deployment Pattern
* **Pattern:** Real-time REST API
* **Justification:** Homebuyers and brokers require sub-second interactive valuations. Real-time predictions via HTTP request-response fulfill this operational criterion.

## 2. Requirements & Setup
Install dependencies via pip:
```bash
pip install -r requirements.txt
```

Run the API locally:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 3. Containerized Deployment (Docker)
Build the Docker container:
```bash
docker build -t housing-estimation-service .
```
Run the container:
```bash
docker run -p 8000:8000 housing-estimation-service
```

## 4. Endpoints Provided
* `GET /`: Healthcheck endpoint displaying model version and expected features.
* `POST /predict`: Handles validated payload, maps correct feature alignment, returns valuation prediction in USD.
