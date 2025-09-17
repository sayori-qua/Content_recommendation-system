FROM python:3.10-slim

WORKDIR /app

#установим системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

#копируем файлы, если requirements не изменился
COPY requirements.txt .

#обновим pip и установим зависимости, если есть изменения в requirements
# Установим зависимости, указав правильный индекс для PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
#запуск
CMD ["python", "app.py"]