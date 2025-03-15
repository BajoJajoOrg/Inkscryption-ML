import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")  # Путь к модели
    LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
    LOG_FILE = os.path.join(LOG_DIR, "app.log")
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))