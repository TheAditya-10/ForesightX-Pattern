FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY ForesightX-Pattern/requirements.inference.txt /app/requirements.inference.txt
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /app/requirements.inference.txt

COPY ForesightX-Pattern /app

RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/health', timeout=3).read()"

CMD ["uvicorn", "foresightx_pattern.app.main:app", "--host", "0.0.0.0", "--port", "8003"]
