# Stage 1: Build the React Frontend
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Build the FastAPI Backend
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
COPY requirements-llm.txt .
RUN pip install --no-cache-dir -r requirements.txt -r requirements-llm.txt

# Copy backend code
COPY . .

# Copy built frontend from Stage 1 into the backend's directory structure
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

ENV PYTHONPATH=/app

# Expose the API port
EXPOSE 8000

# Start FastAPI and serve both backend routes and frontend static files
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
