# Импорт необходимых библиотек
import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Заголовок приложения
st.title("Классификация изображений")

# Опция для выбора режима ввода изображения
mode = st.radio("Выберите способ ввода изображения:", ("Нарисовать изображение", "Загрузить изображение"))

# Функция для отправки изображения на сервер и получения предсказания
def get_prediction(image_data):
    """
    Отправляет изображение на сервер FastAPI и возвращает предсказание.
    """
    try:
        # Отправка POST-запроса с изображением
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": ("image.png", image_data, "image/png")}
        )
        response.raise_for_status()
        data = response.json()
        # Получаем predicted_class вместо prediction
        return data["predicted_class"]
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при отправке запроса: {e}")
    except ValueError as e:
        st.error(f"Ошибка при обработке ответа: {e}")
    return None

# Функция предобработки изображения перед отправкой на сервер
def preprocess_image_client(image):
    """
    Преобразует изображение в нужный формат перед отправкой
    """
    # Преобразуем в оттенки серого, если это цветное изображение
    if image.mode == 'RGBA':
        image = image.convert('L')
    elif image.mode == 'RGB':
        image = image.convert('L')

    # Изменяем размер до 28x28
    image = image.resize((28, 28), Image.LANCZOS)

    return image

# Обработка режима "Загрузить изображение"
if mode == "Загрузить изображение":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            # Показываем оригинальное изображение
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Загруженное изображение", use_container_width=True)


            # Предобрабатываем изображение перед отправкой
            processed_image = preprocess_image_client(original_image)

            # Для отладки можно показать обработанное изображение
            # st.image(processed_image, caption="Обработанное изображение (28x28)", width=56)

            # Преобразование в байты для отправки
            img_byte_arr = io.BytesIO()
            processed_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Получение предсказания
            prediction = get_prediction(img_byte_arr)


        if prediction is not None:
            with col2:
                st.markdown(
                    f"<div style='display: flex; align-items: center; justify-content: center; height: 100%;'>"
                    f"<h1 style='font-size: 48px;'>{prediction}</h1>"
                    f"</div>",
                    unsafe_allow_html=True
                )    


# Обработка режима "Нарисовать изображение"
elif mode == "Нарисовать изображение":
    stroke_width = st.slider("Толщина линии:", 1, 25, 9)

    col_color1, col_color2 = st.columns(2)
    with col_color1:
        stroke_color = st.color_picker("Цвет линии:", "#FFFFFF")
    with col_color2:
        bg_color = st.color_picker("Цвет фона:", "#000000")

    realtime_update = st.checkbox("Обновлять в реальном времени", True)

    col1, col2 = st.columns(2)

    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            update_streamlit=realtime_update,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )     

        if canvas_result.image_data is not None:
            # Преобразование данных холста в изображение
            original_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')

            # Предобрабатываем изображение перед отправкой
            processed_image = preprocess_image_client(original_image)

            # Преобразование в байты для отправки
            img_byte_arr = io.BytesIO()
            processed_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Получение предсказания
            prediction = get_prediction(img_byte_arr)   
        
            if prediction is not None:
                with col2:
                    st.markdown(
                        f"<div style='display: flex; align-items: center; justify-content: center; height: 280px;'>"
                        f"<h1 style='font-size: 48px;'>{prediction}</h1>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

