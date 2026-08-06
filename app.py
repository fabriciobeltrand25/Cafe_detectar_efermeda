import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from openai import OpenAI
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuración principal de la página
st.set_page_config(
    page_title="AgroDetect v2.0 - Diagnóstico Foliar de Café",
    page_icon="🌿",
    layout="wide"
)

# ----------------------------------------------------
# CARGA DE MODELO Y CONFIGURACIÓN
# ----------------------------------------------------
MODEL_PATH = os.path.join("model", "agrodetect_mobilenetv2.keras")
CONFIG_PATH = os.path.join("model", "config_app.json")

CLASES_DEFAULT = ["cercospora", "minador", "phoma", "roya", "sana"]
IMG_SIZE_DEFAULT = (224, 224)

@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"No se encontró el modelo en {MODEL_PATH}. Verifique la carpeta /model en GitHub.")
        return None

@st.cache_data(show_spinner=False)
def cargar_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        clases = config.get("clases", CLASES_DEFAULT)
        img_size = tuple(config.get("img_size", IMG_SIZE_DEFAULT))
        mapa_raw = config.get("mapa_enfermedades", {})
        mapa = {int(k): str(v) for k, v in mapa_raw.items()} if mapa_raw else {}
        return clases, img_size, mapa
    return CLASES_DEFAULT, IMG_SIZE_DEFAULT, {}

modelo = cargar_modelo()
clases, img_size, mapa_enfermedades = cargar_config()

# Metadata descriptiva para la interfaz
INFO_ENFERMEDADES = {
    "roya": {"nombre": "Roya del Café", "cientifico": "Hemileia vastatrix", "severidad": "Alta"},
    "cercospora": {"nombre": "Cercospora / Mancha de Hierro", "cientifico": "Cercospora coffeicola", "severidad": "Media - Alta"},
    "phoma": {"nombre": "Phoma / Quema o Derretimiento", "cientifico": "Phoma costarricensis", "severidad": "Alta en Zonas Altas"},
    "minador": {"nombre": "Minador de la Hoja", "cientifico": "Leucoptera coffeella", "severidad": "Media"},
    "sana": {"nombre": "Hoja Sana", "cientifico": "Coffea arabica / canephora", "severidad": "Ninguna"}
}

# ----------------------------------------------------
# INTERFAZ GRÁFICA DE USUARIO
# ----------------------------------------------------
st.title("🌿 AgroDetect v2.0 - Detección de Enfermedades en Café")
st.caption("Servicio Web Basado en Inteligencia Artificial y Diagnóstico Actuarial/Agrónomo")

col_izq, col_der = st.columns([1, 1])

with col_izq:
    st.subheader("1. Captura o Carga de Imagen")
    metodo = st.radio("Fuente de la imagen:", ["Subir archivo", "Usar cámara"], horizontal=True)
    
    imagen_pil = None
    if metodo == "Subir archivo":
        archivo = st.file_uploader("Seleccione una imagen foliar...", type=["jpg", "jpeg", "png"])
        if archivo:
            imagen_pil = Image.open(archivo)
    else:
        foto = st.camera_input("Tome una foto a la hoja")
        if foto:
            imagen_pil = Image.open(foto)

    if imagen_pil:
        st.image(imagen_pil, caption="Imagen seleccionada", use_column_width=True)

with col_der:
    st.subheader("2. Diagnóstico y Análisis")
    
    if imagen_pil and modelo:
        # Preprocesamiento de la imagen
        img_rgb = imagen_pil.convert("RGB")
        img_resized = img_rgb.resize(img_size)
        img_array = np.array(img_resized)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)

        # Predicción del modelo
        predicciones = modelo.predict(img_preprocessed, verbose=0)[0]
        idx_clase = np.argmax(predicciones)
        confianza = float(predicciones[idx_clase]) * 100
        
        # Mapeo seguro de la clase
        clase_detectada = mapa_enfermedades.get(idx_clase, clases[idx_clase])
        info = INFO_ENFERMEDADES.get(clase_detectada, {"nombre": clase_detectada.upper(), "cientifico": "N/A", "severidad": "N/A"})

        # Mostrar Resultados
        st.markdown(f"### **Resultado:** {info['nombre']}")
        st.write(f"**Nombre Científico:** *{info['cientifico']}*")
        st.write(f"**Nivel de Severidad:** {info['severidad']}")
        st.metric(label="Porcentaje de Confianza", value=f"{confianza:.2f}%")

        st.divider()

        # ----------------------------------------------------
        # INTEGRACIÓN API GROQ
        # ----------------------------------------------------
        st.subheader("💡 Recomendación Técnica (IA Groq)")
        
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

        if api_key:
            with st.spinner("Generando informe técnico con Groq..."):
                prompt = f"""
                Actúa como un agrónomo especialista en el cultivo de café.
                Proporciona un informe técnico estructurado para la condición detectada:

                Condición: {info['nombre']} ({info['cientifico']})
                Porcentaje de confianza: {confianza:.2f}%
                Severidad: {info['severidad']}

                Por favor incluye:
                1. Descripción breve de la enfermedad o condición.
                2. Recomendaciones técnicas para el manejo preventivo y correctivo.
                3. Buenas prácticas para el cuidado del cultivo.
                4. Acciones de seguimiento y monitoreo recomendado.
                """

                try:
                    client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.groq.com/openai/v1"
                    )

                    respuesta = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": "Eres un asesor técnico agrónomo profesional experto en caficultura."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )

                    st.info(respuesta.choices[0].message.content)

                except Exception as e:
                    st.warning(f"Error al conectar con la API de Groq: {e}")
        else:
            st.warning("⚠️ Agregue su `GROQ_API_KEY` en los Secretos de Streamlit Cloud o archivo `.streamlit/secrets.toml` para activar el informe agrónomo.")

    elif not imagen_pil:
        st.info("Por favor cargue o tome una fotografía de una hoja para realizar el diagnóstico.")

st.divider()
st.caption("Servicio Web de Computación en la Nube | UTH 2026")