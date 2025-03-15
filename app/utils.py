import logging
import os
import requests
from PIL import Image
from io import BytesIO
from fastapi import HTTPException
from app.config import Config

# Настройка логирования
if not os.path.exists(Config.LOG_DIR):
    os.makedirs(Config.LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_image_from_url(image_url: str) -> Image.Image:
    """Загрузка изображения по URL."""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        logger.info(f"Изображение успешно загружено по URL: {image_url}")
        return image
    except requests.RequestException as e:
        logger.error(f"Ошибка загрузки изображения по URL {image_url}: {str(e)}")
        raise HTTPException(status_code=400, detail="Не удалось загрузить изображение")
    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {str(e)}")
        raise HTTPException(status_code=400, detail="Некорректное изображение")