from fastapi import HTTPException
from app.utils import logger
import time
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

class MLModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Загрузка модели (заглушка)."""
        try:
            logger.info(f"Загружаем модель из {self.model_path}")
            self.processor = TrOCRProcessor.from_pretrained(self.model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise Exception("Не удалось загрузить модель")

    def predict(self, image_url: str) -> str:
        """Предсказание текста на основе изображения."""
        try:
            logger.info("Выполняется предсказание")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = self.model
            processor = self.processor
            model.to(device)
            
            # Модель в режиме оценки
            model.eval()
            
            # Подготовка изображения
            #image_path = "test0.png"
            #image = Image.open(image_path).convert("RGB")
            #image_url = "https://storage.googleapis.com/kagglesdsdata/datasets/1502872/3977616/test/test0.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250315%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250315T114815Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=9c0d7afd4dbe2663906e764d788959e89daa6b60258fd0444c676e99eb1d3af1dae1bbb1722e1d72c4d9935777b35c2cc58bf8c0c5b3ef2da7bf2c91266b62c35683f7cfdfd3821e54650641dbb9abb8183d8e696fe1a86bad79921d807e9da15439b6daa687587624c3a2b124e8c964ccd5969e57e7201d2a6b82a5f7dcd6a2acb5fbe655e6d19ede0f8ac159a29e7b9388957e667199cf3b7b58192451a22ae6498d0db76ab20e7ec80415f62d9978084bda7c530406203119317b9a867bd957f3269d77da1ff0c31a0a6e071f932221ce27eecfacec68250a6904caf6233bc08a4ad223b9b2276b7400ce17538d9327a570dec946ec75987499ac36f1f694"
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Предобработка изображения с помощью процессора
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)  # Переносим на GPU
            
            # Замер времени начала
            start_time = time.time()
            
            # Генерация текста
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            
            # Декодирование результата
            predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Замер времени окончания
            #end_time = time.time()
            #elapsed_time = end_time - start_time
            
            #print("Распознанный текст:", predicted_text)
            #print(f"Время распознавания: {elapsed_time:.4f} секунд")

            return predicted_text
        except Exception as e:
            logger.error(f"Ошибка предсказания: {str(e)}")
            raise HTTPException(status_code=500, detail="Ошибка модели")