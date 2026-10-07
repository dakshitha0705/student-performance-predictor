FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the project. .dockerignore keeps datasets, logs, MLflow runs and model artifacts OUT of the image;
# artifacts are mounted read-only at runtime and logs are mounted writable (see compose.yaml).
COPY . .

EXPOSE 8000 8501

# Default command = the API. compose.yaml sets the command explicitly for each service.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
