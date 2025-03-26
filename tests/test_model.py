import unittest
import os, sys
import pandas as pd
from PIL import Image
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r".\src\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r".\src\Tesseract-OCR\tessdata"

from app.model import MLModel
from app.utils import fetch_image_from_url


TEST_URL = "https://ekspertiza-reshenie.ru/upload/medialibrary/901/901f80b85cc5194d04bcc5169f00b02f.png"
TEST_IMG = os.path.join(ROOT_DIR, "tests", "simple16.png")

TSV_FILE = os.path.join(ROOT_DIR, "data/test", "test.tsv")
DATA_DIR = os.path.join(ROOT_DIR, "data", "test")
ERRORS_FILE = os.path.join(ROOT_DIR, "data", "errors.csv")

class TestMLModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Настраиваем тест перед выполнением всех тестов"""
        cls.model = MLModel(model_path="raxtemur/trocr-base-ru")

    def test_load_model(self):
        """Проверка загрузки модели"""
        self.assertIsNotNone(self.model.processor, "Процессор не загружен")
        self.assertIsNotNone(self.model.model, "Модель не загружена")

    def test_parse_url(self):
        """Проверка парсинга фотографий"""
        img = fetch_image_from_url(TEST_URL)
        self.assertIsInstance(img,Image.Image, 'Не удалось спарсить изображение')
    
    def test_prediction(self):
        """Тест предсказания"""
        img = Image.open(TEST_IMG).convert("RGB")
        predicted_text = self.model.predict(img)
        
        self.assertEqual(predicted_text, "Ну что, получилось Ли чего-нибудь добиться. Вроде хорошо", "Предсказанный текст не соответствует ожидаемому")

    def test_metrics_calculation(self):
        """Тест расчёта метрик"""
        df = pd.read_csv(TSV_FILE, sep="\t", names=["filename", "text"])
        image_files = df["filename"].tolist()
        texts = df["text"].tolist()
        
        images = []
        references = []
        
        for filename, text in zip(image_files, texts):
            image_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                references.append(text)
            else:
                print(f"Пропущено: {image_path} не найдено")
        
        metrics = self.model.calculate_metrics(images, references)
        
        print("Метрики OCR-модели:")
        error_count = 0
        for key, value in metrics.items():
            if key == "Errors":
                print("Errors")
                for error in metrics[key]:
                    if error_count < 20: 
                        for key2,value2 in error.items():
                            print(f"{key2}: {value2}")
                    else:
                        break
                    error_count+=1
                    
            else:
                print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    unittest.main()
