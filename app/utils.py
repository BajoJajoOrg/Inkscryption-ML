import logging
import os
from PIL import Image
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

def process_image(image_data: bytes) -> Image.Image:
    """Преобразование байтов в изображение с обработкой ошибок."""
    try:
        image = Image.open(io.BytesIO(image_data))
        logger.info("Изображение успешно обработано")
        return image
    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {str(e)}")
        raise HTTPException(status_code=400, detail="Некорректное изображение")