import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Загрузка модели
model_path = "best_classification_model.h5"
model = tf.keras.models.load_model(model_path, compile=False)

def preprocess_image(image):
    # Преобразование в оттенки серого, если цветное изображение
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = np.mean(image, axis=2)

    # Изменение размера до 28x28
    image = tf.image.resize(tf.expand_dims(image, -1), [28, 28])

    # Нормализация значений пикселей [0, 1]
    image = image / 255.0

    # Добавление размерности батча
    image = tf.expand_dims(image, 0)

    return image


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Чтение содержимого файла
    contents = await file.read()

    # Открытие изображения
    image = Image.open(io.BytesIO(contents))
    image_array = np.array(image)

    # Предобработка изображения
    processed_image = preprocess_image(image_array)

    # Получение предсказаний
    predictions = model.predict(processed_image)
    predicted_class = int(np.argmax(predictions[0]))

    # Вычисление вероятностей для каждого класса
    probabilities = [float(p) for p in predictions[0]]

    return JSONResponse(content={
        "predicted_class": predicted_class,
        "probabilities": probabilities
    })