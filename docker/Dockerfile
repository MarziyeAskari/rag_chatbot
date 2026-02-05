FROM python:3.11-slim

WORKDIR /app

# Better logs in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps (only if you need to compile wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install your project (pyproject.toml based)
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "app.run"]

