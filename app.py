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

/* ---- ESTILOS PARA LOS PASOS ---- */
.step-container {
    margin-top: 20px;
}

.step {
    display: flex;
    align-items: flex-start;
    margin-bottom: 22px;
    padding-bottom: 22px;
    border-bottom: 1px solid #E2D7CC;
}

.step:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}

.step-number {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: #1E5B43;
    display: flex;
    justify-content: center;
    align-items: center;
    font-weight: 800;
    color: white;
    margin-right: 18px;
    font-size: 14px;
    flex-shrink: 0;
    letter-spacing: 0.5px;
}

.step-title {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 6px;
    color: #2B1A10;
}

.step-text {
    font-size: 14px;
    color: #4A3C33;
    line-height: 1.8;
}

.step-orientation-title {
    font-size: 20px;
    font-weight: 700;
    color: #2A1A10;
    margin-bottom: 8px;
    margin-top: 10px;
}

.step-orientation-subtitle {
    font-size: 15px;
    color: #4A3C33;
    margin-bottom: 20px;
}
/* ---- FIN ESTILOS PASOS ---- */

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
# INFORMACIÓN DE ENFERMEDADES + RECOMENDACIONES
# -------------------------------------------------------

INFO_ENFERMEDADES = {
    "roya": {
        "nombre": "Roya del Café",
        "cientifico": "Hemileia vastatrix",
        "severidad": "Alta",
        "pasos": [
            {
                "titulo": "Diferenciación a simple vista",
                "texto": "Roya anaranjada (Hemileia vastatrix) produce pústulas de color naranja-amarillo en el envés de la hoja, que al frotarse liberan polvo anaranjado (esporas). Se diferencia de Cercospora por la ausencia de halos definidos y por su distribución en manchas irregulares que coalescen."
            },
            {
                "titulo": "Manejo agronómico preventivo y correctivo",
                "texto": "Mantener sombra moderada (40-50%) para reducir estrés hídrico. Aplicar fungicidas sistémicos (triazoles o estrobilurinas) en épocas de alta humedad. Realizar podas sanitarias para eliminar hojas infectadas. Evitar exceso de nitrógeno que favorece tejido tierno susceptible."
            },
            {
                "titulo": "Consulta a un técnico IHCAFE",
                "texto": "Consulte si la infección supera el 20% del follaje o si aparecen brotes en ramas bajas. El técnico evaluará la necesidad de fungicidas específicos y ajustará el plan de manejo integrado."
            },
            {
                "titulo": "Monitoreo y seguimiento",
                "texto": "Monitoree semanalmente en épocas lluviosas (mayo-noviembre). Revise el tercio inferior de la planta donde inicia la infección. Mejora: disminución de pústulas y brotes verdes. Empeora: defoliación y pérdida de producción."
            },
            {
                "titulo": "Registro y trazabilidad",
                "texto": "Registre incidencia semanal, condiciones climáticas (humedad, temperatura), aplicaciones fungicidas y resultados. Documente si la parcela está en zonas bajas o con alta humedad relativa."
            }
        ]
    },
    "cercospora": {
        "nombre": "Cercospora / Mancha de Hierro",
        "cientifico": "Cercospora coffeicola",
        "severidad": "Media - Alta",
        "pasos": [
            {
                "titulo": "Diferenciación a simple vista",
                "texto": "Cercospora coffeicola produce manchas circulares de 3-8 mm con centro grisáceo-necrótico y halo amarillo-anaranjado. A diferencia de la Roya, no hay pústulas en relieve ni esporas en el envés. Se confunde frecuentemente con manchas de nutrición (deficiencia de Mn), pero estas carecen del halo definido y son más irregulares."
            },
            {
                "titulo": "Manejo agronómico preventivo y correctivo",
                "texto": "Aumentar fertilización nitrogenada (urea foliar al 2%) y potásica. Regular sombra al 40-50% para reducir estrés hídrico. Aplicar caldo bordelés preventivo antes de lluvias intensas. En brotes activos, usar fungicidas específicos con mancozeb o clorotalonil. Evitar trabajos en campo con follaje mojado para no diseminar esporas."
            },
            {
                "titulo": "Consulta a un técnico IHCAFE",
                "texto": "Consulte si las manchas aparecen en más del 30% del follaje o si persisten tras dos aplicaciones fungicidas. El técnico analizará niveles de N y K en suelo/hoja para descartar que sea un problema nutricional primario que debilite el tejido y facilite la infección."
            },
            {
                "titulo": "Monitoreo y seguimiento",
                "texto": "Monitoree quincenalmente en épocas secas y calurosas (febrero-abril), cuando el estrés hídrico potencia la enfermedad. Revise hojas del tercio medio de la planta. Mejora: nuevas hojas sin manchas, recuperación del color verde intenso. Empeora: coalescencia de manchas, secado de bordes foliares."
            },
            {
                "titulo": "Registro y trazabilidad",
                "texto": "Registre análisis foliar bianuales, niveles de sombra (% cobertura), tipo de sombra (Inga, Erythrina, malla), fecha de aplicaciones y condiciones climáticas previas. Documente si la parcela está en ladera expuesta al sol (mayor riesgo). Estos datos son clave para ajustar el manejo integral del cultivo."
            }
        ]
    },
    "phoma": {
        "nombre": "Phoma / Derretimiento",
        "cientifico": "Phoma costarricensis",
        "severidad": "Alta",
        "pasos": [
            {
                "titulo": "Diferenciación a simple vista",
                "texto": "Phoma costarricensis produce manchas necróticas de forma irregular que avanzan desde el borde de la hoja hacia el centro, con un color marrón oscuro. En ataques severos causa 'derretimiento' de hojas tiernas. Se diferencia de la Roya por la ausencia de pústulas y su avance desde el margen foliar."
            },
            {
                "titulo": "Manejo agronómico preventivo y correctivo",
                "texto": "Mantener buena aireación entre plantas. Evitar riego por aspersión. Aplicar fungicidas protectores (clorotalonil) en épocas de alta humedad. En ataques severos, usar fungicidas sistémicos (tebuconazol). Realizar podas de renovación en ramas afectadas."
            },
            {
                "titulo": "Consulta a un técnico IHCAFE",
                "texto": "Consulte si el daño supera el 15% del follaje o si afecta ramas productivas. El técnico evaluará la necesidad de aplicaciones específicas y ajustará el programa de fertilización."
            },
            {
                "titulo": "Monitoreo y seguimiento",
                "texto": "Monitoree semanalmente en épocas lluviosas. Revise brotes tiernos y hojas del tercio superior. Mejora: detención del avance de manchas, brotes sanos. Empeora: defoliación y muerte de ramas."
            },
            {
                "titulo": "Registro y trazabilidad",
                "texto": "Registre condiciones climáticas, densidad de siembra, aplicaciones fungicidas y severidad del daño. Documente si hay historial de la enfermedad en la parcela."
            }
        ]
    },
    "minador": {
        "nombre": "Minador de la Hoja",
        "cientifico": "Leucoptera coffeella",
        "severidad": "Media",
        "pasos": [
            {
                "titulo": "Diferenciación a simple vista",
                "texto": "Leucoptera coffeella produce minas (galerías) serpentiformes en el haz de la hoja, de color marrón claro. Las larvas se alimentan del parénquima. Se diferencia de enfermedades fúngicas por la forma de galería y la ausencia de esporas o pústulas."
            },
            {
                "titulo": "Manejo agronómico preventivo y correctivo",
                "texto": "Mantener sombra adecuada para reducir población de la plaga. Aplicar insecticidas específicos (lambda-cialotrina) en épocas de brotación. Realizar control biológico con parasitoides (Trichogramma)."
            },
            {
                "titulo": "Consulta a un técnico IHCAFE",
                "texto": "Consulte si más del 40% de las hojas presentan minas. El técnico evaluará la necesidad de control químico y ajustará el manejo integrado de plagas."
            },
            {
                "titulo": "Monitoreo y seguimiento",
                "texto": "Monitoree semanalmente en épocas secas. Revise hojas del tercio superior. Mejora: disminución de minas nuevas. Empeora: aumento de galerías y defoliación."
            },
            {
                "titulo": "Registro y trazabilidad",
                "texto": "Registre población de la plaga, aplicaciones insecticidas, y resultados. Documente si hay presencia de parasitoides naturales."
            }
        ]
    },
    "sana": {
        "nombre": "Hoja Sana",
        "cientifico": "Coffea arabica",
        "severidad": "Sin Riesgo",
        "pasos": [
            {
                "titulo": "Condición de la planta",
                "texto": "La hoja presenta un color verde intenso y uniforme, sin manchas, pústulas o galerías. El tejido foliar está firme y sin daños visibles."
            },
            {
                "titulo": "Manejo preventivo",
                "texto": "Mantener prácticas agronómicas adecuadas: fertilización balanceada, manejo de sombra, y riego controlado. La prevención es clave para mantener el cultivo sano."
            },
            {
                "titulo": "Monitoreo continuo",
                "texto": "Realice monitoreos regulares para detectar tempranamente cualquier cambio en el follaje. La detección temprana es fundamental para el manejo efectivo."
            },
            {
                "titulo": "Registro de condiciones",
                "texto": "Documente condiciones climáticas, prácticas de manejo y estado del cultivo para mantener trazabilidad."
            },
            {
                "titulo": "Seguimiento",
                "texto": "Continúe con el monitoreo quincenal para asegurar la detección temprana de cualquier patología."
            }
        ]
    }
}

# -------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------

if "historial" not in st.session_state:
    st.session_state.historial = []

# -------------------------------------------------------
# FUNCIÓN PARA GENERAR PASOS HTML
# -------------------------------------------------------

def generar_pasos_html(pasos):
    html = '<div class="step-container">'
    
    # Títulos de orientación
    html += '''
    <div class="step-orientation-title">📋 Orientación y manejo preventivo</div>
    <div class="step-orientation-subtitle">Aquí tienes una recomendación técnica detallada para manejar la situación:</div>
    '''
    
    for i, paso in enumerate(pasos, 1):
        html += f'''
        <div class="step">
            <div class="step-number">{i:02d}</div>
            <div>
                <div class="step-title">{paso["titulo"]}</div>
                <div class="step-text">{paso["texto"]}</div>
            </div>
        </div>
        '''
    html += '</div>'
    return html

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
    pasos = []
    
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
            "severidad": "N/A",
            "pasos": []
        })
        
        nombre = info["nombre"]
        cientifico = info["cientifico"]
        severidad = info["severidad"]
        pasos = info.get("pasos", [])
        diagnostico_generado = True
        
        # Actualizar historial
        st.session_state.historial.insert(0, nombre)
        st.session_state.historial = st.session_state.historial[:5]
    
    # ---------------------------------------
    # MOSTRAR RESULTADO (DENTRO DEL RESULT-BOX)
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
    # PANEL RESULT-BOX (TODO DENTRO)
    # ---------------------------------------
    
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    if diagnostico_generado:
        st.success(f"✅ Nivel de severidad detectado: {severidad}")
        
        # --- LÍNEA SEPARADORA ---
        st.markdown("---")
        
        # --- ORIENTACIÓN Y MANEJO PREVENTIVO (DENTRO DEL RESULT-BOX) ---
        if pasos:
            # Generar y mostrar pasos en formato HTML
            pasos_html = generar_pasos_html(pasos)
            st.markdown(pasos_html, unsafe_allow_html=True)
            
    else:
        st.info("Cargue una imagen para comenzar el diagnóstico.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------------------------------------
    # HISTORIAL RECIENTE (FUERA DEL RESULT-BOX)
    # ---------------------------------------
    
    st.markdown("<br>", unsafe_allow_html=True)
    
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
        from datetime import datetime
        fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M")
        for i, item in enumerate(st.session_state.historial):
            if i == 0:
                st.markdown(f"""
                <div class="history">
                    <strong>🟤 {item}</strong><br>
                    <span style="font-size:12px; color:#8B8076;">{fecha_actual}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
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
