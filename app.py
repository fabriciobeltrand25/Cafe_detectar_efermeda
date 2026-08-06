import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from openai import OpenAI
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------------------------------
# CONFIGURACIÓN PRINCIPAL
# -------------------------------------------------------

st.set_page_config(
    page_title="Cafe Detecter - Diagnóstico Foliar",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# CSS PERSONALIZADO
# -------------------------------------------------------

st.markdown("""
<style>

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

html, body, [data-testid="stAppViewContainer"] {
    background: #F5F1EB;
    font-family: Inter, sans-serif;
}

.block-container {
    padding-top: 25px;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1600px;
}

h1, h2, h3, h4 {
    color: #2A1A10;
}

.big-title {
    font-size: 46px;
    font-weight: 800;
    color: #2F170D;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 14px;
    color: #7B6D62;
    margin-bottom: 25px;
}

.upload-box {
    background: white;
    border-radius: 20px;
    padding: 20px;
    border: 2px dashed #DDD5CC;
}

.result-box {
    background: #EEE6DD;
    padding: 30px;
    border-radius: 22px;
    border: 1px solid #DED3C8;
}

.info-card {
    background: white;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,.04);
    margin-top: 15px;
}

.metric-title {
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8A7D70;
    font-weight: 700;
}

.metric-big {
    font-size: 52px;
    font-weight: 800;
    color: #2A1A10;
}

.disease-title {
    font-size: 42px;
    font-weight: 800;
    color: #2A1A10;
    margin-top: 10px;
    margin-bottom: 10px;
}

.scientific {
    font-style: italic;
    font-size: 14px;
    color: #75685C;
    margin-bottom: 25px;
}

.history {
    background: white;
    border-radius: 15px;
    padding: 14px;
    margin-top: 20px;
    border: 1px solid #E2D7CC;
}

.footer {
    text-align: center;
    font-size: 13px;
    color: #8B8076;
    margin-top: 30px;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    background: #2A1A10;
    color: white;
    border: none;
    padding: 12px;
    font-weight: 700;
}

.stButton>button:hover {
    background: #3C261A;
    color: white;
}

[data-testid="stFileUploader"] {
    border: none;
}

img {
    border-radius: 16px;
}

hr {
    margin-top: 35px;
    margin-bottom: 35px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# RUTAS DE MODELO Y CONFIGURACIÓN
# -------------------------------------------------------

MODEL_PATH = os.path.join("model", "agrodetect_mobilenetv2.keras")
CONFIG_PATH = os.path.join("model", "config_app.json")

CLASES_DEFAULT = ["cercospora", "minador", "phoma", "roya", "sana"]
IMG_SIZE_DEFAULT = (224, 224)

# -------------------------------------------------------
# CARGA DE MODELO
# -------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        st.error(f"No se encontró el modelo en {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# -------------------------------------------------------
# CARGA DE CONFIGURACIÓN
# -------------------------------------------------------

@st.cache_data(show_spinner=False)
def cargar_config():
    if not os.path.exists(CONFIG_PATH):
        return CLASES_DEFAULT, IMG_SIZE_DEFAULT, {}
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    clases = cfg.get("clases", CLASES_DEFAULT)
    img_size = tuple(cfg.get("img_size", IMG_SIZE_DEFAULT))
    mapa_raw = cfg.get("mapa_enfermedades", {})
    mapa = {int(k): str(v) for k, v in mapa_raw.items()}
    
    return clases, img_size, mapa

modelo = cargar_modelo()
clases, img_size, mapa_enfermedades = cargar_config()

# -------------------------------------------------------
# INFORMACIÓN DE ENFERMEDADES
# -------------------------------------------------------

INFO_ENFERMEDADES = {
    "roya": {
        "nombre": "Roya del Café",
        "cientifico": "Hemileia vastatrix",
        "severidad": "Alta"
    },
    "cercospora": {
        "nombre": "Cercospora / Mancha de Hierro",
        "cientifico": "Cercospora coffeicola",
        "severidad": "Media - Alta"
    },
    "phoma": {
        "nombre": "Phoma / Derretimiento",
        "cientifico": "Phoma costarricensis",
        "severidad": "Alta"
    },
    "minador": {
        "nombre": "Minador de la Hoja",
        "cientifico": "Leucoptera coffeella",
        "severidad": "Media"
    },
    "sana": {
        "nombre": "Hoja Sana",
        "cientifico": "Coffea arabica",
        "severidad": "Sin Riesgo"
    }
}

# -------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------

if "historial" not in st.session_state:
    st.session_state.historial = []

# -------------------------------------------------------
# CABECERA PRINCIPAL
# -------------------------------------------------------

st.markdown("""
<div class="big-title">
🌿 Captura de Imagen Foliar
</div>

<div class="subtitle">
Posicione la hoja de café bajo luz natural. El sistema detectará automáticamente signos de Roya, Cercospora, Phoma, Minador o una hoja sana.
</div>
""", unsafe_allow_html=True)

# ======================================================
# INTERFAZ PRINCIPAL - DOS COLUMNAS
# ======================================================

col_izq, col_der = st.columns([1.08, 1], gap="large")

# ======================================================
# COLUMNA IZQUIERDA - CARGA DE IMAGEN
# ======================================================

with col_izq:
    st.markdown("### 📷 Captura de Imagen")
    
    metodo = st.radio(
        "",
        ["📁 Subir archivo", "📷 Usar cámara"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    imagen_pil = None
    
    if metodo == "📁 Subir archivo":
        archivo = st.file_uploader(
            "Seleccione una imagen",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if archivo is not None:
            imagen_pil = Image.open(archivo).convert("RGB")
    else:
        foto = st.camera_input("Tomar fotografía")
        if foto is not None:
            imagen_pil = Image.open(foto).convert("RGB")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if imagen_pil is not None:
        st.image(imagen_pil, caption="Imagen seleccionada", use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-box">
            <br><br><br>
            <center>
                <h3>☁️</h3>
                <h4>No se ha seleccionado ninguna imagen</h4>
                <p>Suba una fotografía de una hoja de café para comenzar el análisis.</p>
                <br><br><br>
            </center>
        </div>
        """, unsafe_allow_html=True)

# ======================================================
# COLUMNA DERECHA - RESULTADOS
# ======================================================

with col_der:
    st.markdown("""
    <div class="metric-title">
    ÚLTIMO DIAGNÓSTICO
    </div>
    """, unsafe_allow_html=True)
    
    # Variables por defecto
    nombre = "--"
    cientifico = "--"
    severidad = "--"
    confianza = 0.0
    diagnostico_generado = False
    clase = None
    
    # ---------------------------------------
    # PROCESAR IMAGEN CON EL MODELO
    # ---------------------------------------
    
    if imagen_pil is not None and modelo is not None:
        img_rgb = imagen_pil.convert("RGB")
        img_resized = img_rgb.resize(img_size)
        img_array = np.array(img_resized)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        
        predicciones = modelo.predict(img_preprocessed, verbose=0)[0]
        idx_clase = np.argmax(predicciones)
        confianza = float(predicciones[idx_clase]) * 100
        
        clase = mapa_enfermedades.get(idx_clase, clases[idx_clase])
        info = INFO_ENFERMEDADES.get(clase, {
            "nombre": clase.upper(),
            "cientifico": "N/A",
            "severidad": "N/A"
        })
        
        nombre = info["nombre"]
        cientifico = info["cientifico"]
        severidad = info["severidad"]
        diagnostico_generado = True
        
        # Actualizar historial
        st.session_state.historial.insert(0, nombre)
        st.session_state.historial = st.session_state.historial[:5]
    
    # ---------------------------------------
    # MOSTRAR RESULTADO
    # ---------------------------------------
    
    c1, c2 = st.columns([4, 1])
    
    with c1:
        st.markdown(f"""
        <div class="disease-title">{nombre}</div>
        <div class="scientific">{cientifico}</div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div style="text-align:right">
            <div class="metric-big">{confianza:.1f}%</div>
            <div class="metric-title">CONFIANZA</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ---------------------------------------
    # PANEL DE SEVERIDAD
    # ---------------------------------------
    
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    if diagnostico_generado:
        st.success(f"Nivel de severidad detectado: {severidad}")
    else:
        st.info("Cargue una imagen para comenzar el diagnóstico.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---------------------------------------
    # INTEGRACIÓN API GROQ (Recomendaciones técnicas)
    # ---------------------------------------
    
    if diagnostico_generado:
        st.markdown("### 💡 Recomendación Técnica")
        
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        
        if api_key:
            with st.spinner("Generando informe técnico con IA..."):
                prompt = f"""
                Actúa como un agrónomo especialista en el cultivo de café.
                Proporciona un informe técnico estructurado para la condición detectada:

                Condición: {nombre} ({cientifico})
                Porcentaje de confianza: {confianza:.2f}%
                Severidad: {severidad}

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
            st.warning("⚠️ Agregue su `GROQ_API_KEY` en los Secretos de Streamlit Cloud para activar el informe agrónomo.")
    
    # ---------------------------------------
    # HISTORIAL RECIENTE
    # ---------------------------------------
    
    st.markdown("""
    <div class="metric-title" style="margin-top: 25px;">
    HISTORIAL RECIENTE
    </div>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.historial) == 0:
        st.markdown("""
        <div class="history">No existen diagnósticos.</div>
        """, unsafe_allow_html=True)
    else:
        for item in st.session_state.historial:
            st.markdown(f"""
            <div class="history">🟤 {item}</div>
            """, unsafe_allow_html=True)
    
    # ---------------------------------------
    # PIE DE PÁGINA
    # ---------------------------------------
    
    st.markdown("""
    <hr>
    <div class="footer">
        ☕ Servicio Web de Diagnóstico Foliar | UTH 2026
    </div>
    """, unsafe_allow_html=True)
