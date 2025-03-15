# Базовый образ
FROM python:3.9-slim

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY ./app /app/app
COPY .env /app/.env

# Указываем порт
EXPOSE 8000

# Команда для запуска
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]