import os
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf

# Шлях до файлу навченої моделі
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.keras"

IMG_SIZE = 100
CLASS_NAMES = ["Кіт", "Собака"]

# Перевірка наявності моделі
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Файл моделі не знайдено: {MODEL_PATH}"
    )

# Завантаження моделі
model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image):
    if image is None:
        return {"Кіт": 0.0, "Собака": 0.0}

    # Перетворення зображення у відтінки сірого
    image = image.convert("RGB").convert("L")

    # Зміна розміру до 100x100 пікселів
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Нормалізація значень пікселів
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    # Підготовка форми масиву для нейронної мережі
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    # Класифікація зображення
    predictions = model.predict(image_array, verbose=0)[0]

    return {
        "Кіт": float(predictions[0]),
        "Собака": float(predictions[1])
    }


# Створення вебінтерфейсу
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(
        type="pil",
        label="Завантажте фотографію"
    ),
    outputs=gr.Label(
        num_top_classes=2,
        label="Результат класифікації"
    ),
    title="Cats Dogs AI",
    description=(
        "Система класифікації зображень котів і собак "
        "на основі згорткової нейронної мережі (CNN)."
    )
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )