import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from fastapi import HTTPException
from app.utils import logger
import Levenshtein
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r".\src\Tesseract-OCR\tesseract.exe"
# import os
# os.environ['TESSDATA_PREFIX'] = r".\src\Tesseract-OCR\tessdata"

class MLModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        """Загрузка процессора и модели."""
        try:
            logger.info(f"Загружаем модель и процессор из {self.model_path}")
            self.processor = TrOCRProcessor.from_pretrained(self.model_path)
            logger.info("Процессор успешно загружен")
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            logger.info("Модель успешно загружена")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Модель переведена на устройство: {self.device}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise Exception("Не удалось загрузить модель")

    def predict(self, image: Image.Image) -> str:
        """Предсказание текста на основе изображения."""
        try:
            logger.info("Выполняется детекция")
            custom_config = r'--psm 6'
            data = pytesseract.image_to_data(image, lang="rus", config=custom_config, output_type=pytesseract.Output.DICT)
            # Группируем bounding boxes по строкам
            line_boxes = {}  # {line_num: (x_min, y_min, x_max, y_max)}
            for i in range(len(data["text"])):
                if data["text"][i].strip():  # Пропускаем пустые строки
                    line_num = data["line_num"][i]  # Номер строки
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    x2, y2 = x + w, y + h

                    if line_num not in line_boxes:
                        line_boxes[line_num] = [x, y, x2, y2]
                    else:
                        line_boxes[line_num][0] = min(line_boxes[line_num][0], x)   # x_min
                        line_boxes[line_num][1] = min(line_boxes[line_num][1], y)   # y_min
                        line_boxes[line_num][2] = max(line_boxes[line_num][2], x2)  # x_max
                        line_boxes[line_num][3] = max(line_boxes[line_num][3], y2)  # y_max
            logger.info("Выполняется предсказание")
            recognized_texts = []
            for (x1, y1, x2, y2) in line_boxes.values():
                cropped = image.crop((x1, y1, x2, y2))
                pixel_values = self.processor(images=[cropped], return_tensors="pt").pixel_values.to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                recognized_texts.append(predicted_text)
            result_text = " ".join(recognized_texts)
            logger.info(f"Предсказанный текст: {result_text}")
            return result_text
        except Exception as e:
            logger.error(f"Ошибка предсказания: {str(e)}")
            # Исправляем синтаксис HTTPException
            raise HTTPException(500, "Ошибка модели")
    
    def test_predict(self, image: Image.Image) -> str:
        """Предсказание текста тестовой выборки с предварительной детекцией."""
        try:
            logger.info("Выполняется предсказание")
            # Добавляем padding=True для корректной обработки
            pixel_values = self.processor(images=image, return_tensors="pt", padding=True).pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info(f"Предсказанный текст: {predicted_text}")
            return predicted_text
        except Exception as e:
            logger.error(f"Ошибка предсказания: {str(e)}")
            # Исправляем синтаксис HTTPException
            raise HTTPException(500, "Ошибка модели")
        
    def calculate_metrics(self, arr_img: list[Image.Image], arr_ref: list[str]) -> dict:
        """
        Вычисляет средние метрики CER и WER по массиву изображений.
        
        :param arr_img: список изображений
        :param arr_ref: список эталонных текстов
        :return: словарь со средними значениями CER, WER и Accuracy
        """
        global_cer = 0
        global_wer = 0
        img_count = 0
        num_samples = len(arr_img)
        errors = []
        
        for img, reference in zip(arr_img, arr_ref):
            pred_str = self.test_predict(img)
            
            global_cer  += Levenshtein.distance(pred_str, reference) / max(len(reference), 1)
            global_wer  += Levenshtein.distance(pred_str.split(), reference.split()) / max(len(reference.split()), 1)
            
            if pred_str != reference:
                errors.append({
                    "true_text": reference,
                    "predicted_text": pred_str,
                    "img_count": img_count
                })
            img_count+=1
        # Усреднение по количеству примеров
        avg_cer = global_cer / num_samples
        avg_wer = global_wer / num_samples
        avg_accuracy = 1 - avg_cer
        
        return {
            "CER": avg_cer,
            "WER": avg_wer,
            "Accuracy": avg_accuracy,
            "Errors": errors
        }