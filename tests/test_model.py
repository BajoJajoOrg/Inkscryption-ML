import unittest
import os, sys
import pandas as pd
from PIL import Image
root_path = os.path.abspath("./Inkscryption-ML/")
sys.path.append(root_path)


from app.model import MLModel
from app.utils import fetch_image_from_url



TEST_URL = 'https://storage.googleapis.com/kagglesdsdata/datasets/1502872/3977616/test/test0.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250315%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250315T114815Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=9c0d7afd4dbe2663906e764d788959e89daa6b60258fd0444c676e99eb1d3af1dae1bbb1722e1d72c4d9935777b35c2cc58bf8c0c5b3ef2da7bf2c91266b62c35683f7cfdfd3821e54650641dbb9abb8183d8e696fe1a86bad79921d807e9da15439b6daa687587624c3a2b124e8c964ccd5969e57e7201d2a6b82a5f7dcd6a2acb5fbe655e6d19ede0f8ac159a29e7b9388957e667199cf3b7b58192451a22ae6498d0db76ab20e7ec80415f62d9978084bda7c530406203119317b9a867bd957f3269d77da1ff0c31a0a6e071f932221ce27eecfacec68250a6904caf6233bc08a4ad223b9b2276b7400ce17538d9327a570dec946ec75987499ac36f1f694'

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TSV_FILE = os.path.join(ROOT_DIR, "data", "test.tsv")
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
        img = fetch_image_from_url(TEST_URL)
        predicted_text = self.model.predict(img)
        
        self.assertEqual(predicted_text, 'ибо', "Предсказанный текст не соответствует ожидаемому")

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
