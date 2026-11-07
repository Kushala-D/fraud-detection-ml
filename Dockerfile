# Use Python 3.11
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy and install runtime dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy API source code and saved model
COPY src/ ./src/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]