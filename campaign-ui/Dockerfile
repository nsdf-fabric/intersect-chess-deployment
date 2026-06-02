FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8080

# Shell form with exec so uvicorn remains PID 1 while ${ROOT_PATH:-} is expanded at runtime.
# Set ROOT_PATH=/campaign (or whatever subpath the reverse proxy uses) when deploying behind a proxy.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --root-path "${ROOT_PATH:-}"
