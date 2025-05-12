# Use Python 3.8 slim image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv (optional, otherwise you just use pip)
# RUN pip install pipenv

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Collect static files (optional: only for production)
# RUN python manage.py collectstatic --noinput

# Expose port (you can change this if needed)
EXPOSE 8000

# Run application using gunicorn (for WSGI apps)
CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8000"]

# --- OR --- for ASGI apps like with Uvicorn (e.g., if you're using Django Channels)
# CMD ["uvicorn", "backend.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
