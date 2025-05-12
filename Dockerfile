# Use official Python 3.9 slim image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application files
COPY . .

# Expose port only if needed (e.g., FastAPI UI)
EXPOSE 8000

# Default command (can be overridden by Kubernetes CronJobs or args)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
