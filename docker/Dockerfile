FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
# If you use requirements.txt instead, replace accordingly.
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e .

COPY . /app

EXPOSE 8000
CMD ["python", "-m", "app.run"]
