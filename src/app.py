import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Ajustar la ruta del modelo usando la ruta absoluta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_transfer.keras')
model = load_model(MODEL_PATH)

# Estilizar la aplicación
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Clasificador de Imágenes: ¿Perro o Gato?</h1>", unsafe_allow_html=True)
st.write("### Sube una imagen de un perro o un gato para predecir su categoría. La aplicación mostrará el resultado abajo.")

# Tipos de archivos permitidos
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image):
    # Procesar la imagen para la predicción
    img = image.resize((100, 100))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    # Realizar la predicción
    pred = model.predict(img)
    return 'Perro' if pred > 0.5 else 'Gato'

# Cargar la imagen en la barra lateral
st.sidebar.markdown("### Opciones de carga")
uploaded_file = st.sidebar.file_uploader("Sube una imagen", type=['png', 'jpg', 'jpeg', 'webp'])

# Separador visual
st.sidebar.markdown("---")

if uploaded_file is not None:
    # Comprobar si la extensión es válida
    if allowed_file(uploaded_file.name):
        # Mostrar la imagen cargada
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen subida.', use_column_width=True)

        # Mostrar separador
        st.markdown("---")
        st.write("### Realizando la predicción...")
        
        # Realizar la predicción
        label = predict_image(image)

        # Mostrar el resultado resaltado
        if label == 'Perro':
            st.success(f'¡Es un **{label}**!')
        else:
            st.info(f'¡Es un **{label}**!')
    else:
        st.warning("Por favor, sube una imagen en formato permitido (png, jpg, jpeg, webp).")
else:
    st.write("Carga una imagen desde la barra lateral para comenzar.")

# Pie de página
st.markdown("<hr style='border-top: 3px solid #FF6347;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Desarrollado con Streamlit y TensorFlow</p>", unsafe_allow_html=True)
