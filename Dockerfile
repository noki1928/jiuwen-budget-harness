FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY run_agent.py ./
COPY jw_harness ./jw_harness

ENTRYPOINT ["python", "/app/run_agent.py"]
