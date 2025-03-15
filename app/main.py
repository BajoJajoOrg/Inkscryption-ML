from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import MLModel
from app.utils import fetch_image_from_url, logger
from app.config import Config

app = FastAPI(title="ML Image-to-Text Service")

# Модель для валидации входных данных
class ImageRequest(BaseModel):
    image_url: str

# Инициализация модели
try:
    ml_model = MLModel(Config.MODEL_PATH)
except Exception as e:
    logger.critical(f"Не удалось запустить сервис: {str(e)}")
    raise e

@app.post("/predict/", response_model=dict)
async def predict(request: ImageRequest):
    """Эндпоинт для предсказания текста по URL изображения."""
    logger.info(f"Получен запрос на предсказание, URL: {request.image_url}")
    
    # Загружаем изображение по URL
    image = fetch_image_from_url(request.image_url)
    
    # Получаем предсказание
    text = ml_model.predict(image)
    
    return {"text": text}

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса."""
    return {"status": "healthy"}