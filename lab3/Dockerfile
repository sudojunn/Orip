# === ЭТАП 1: Сборщик (Builder) ===
FROM python:3.12-slim AS builder
WORKDIR /build
# Устанавливаем системные инструменты компиляции (gcc, g++ и др.)

RUN apt-get update && apt-get install -y --no-install-recommends build-\
essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
# Извлекаем только готовые скомпилированные библиотеки, игнорируя мусор сборщика
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY iris_model.onnx .
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]