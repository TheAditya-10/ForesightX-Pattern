FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY ForesightX-Pattern/requirements.inference.txt /app/requirements.inference.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /app/requirements.inference.txt

COPY ForesightX-Pattern /app

EXPOSE 8003

CMD ["uvicorn", "foresightx_pattern.app.main:app", "--host", "0.0.0.0", "--port", "8003"]
