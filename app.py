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

/* ---- CUADRO PRINCIPAL DE RESULTADO (cafecito claro) ---- */
.result-box {
    background: #EEE6DD;
    padding: 30px;
    border-radius: 22px;
    border: 1px solid #DED3C8;
}

.result-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 18px;
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
    line-height: 1;
}

.disease-title {
    font-size: 38px;
    font-weight: 800;
    color: #2A1A10;
    margin-bottom: 4px;
}

.scientific {
    font-style: italic;
    font-size: 14px;
    color: #75685C;
}

.severity-pill {
    display: inline-block;
    background: #DCEFE1;
    color: #1E5B43;
    border: 1px solid #C3E3CD;
    padding: 10px 16px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 10px;
}

.info-pill {
    display: inline-block;
    background: #E7EEF5;
    color: #2E4E6B;
    border: 1px solid #CFDEEA;
    padding: 10px 16px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 14px;
}

.orientacion-title {
    font-size: 15px;
    font-weight: 700;
    color: #2B1A10;
    margin-top: 22px;
    margin-bottom: 4px;
}

.orientacion-sub {
    font-size: 13px;
    color: #7B6D62;
    margin-bottom: 14px;
}

/* ---- ESTILOS PARA LOS PASOS (dentro del mismo cuadro, sin caja blanca aparte) ---- */
.step-container {
    margin-top: 6px;
}

.step {
    display: flex;
    align-items: flex-start;
    padding: 16px 0;
    border-bottom: 1px solid #DED3C8;
}

.step:last-child {
    border-bottom: none;
    padding-bottom: 0;
}

.step-number {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: #1E5B43;
    display: flex;
    justify-content: center;
    align-items: center;
    font-weight: 800;
    color: white;
    margin-right: 16px;
    font-size: 13px;
    flex-shrink: 0;
    letter-spacing: 0.5px;
}

.step-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 4px;
    color: #2B1A10;
}

.step-text {
    font-size: 13.5px;
    color: #4A3C33;
    line-height: 1.7;
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
# (Sin saltos de línea ni indentación: evita que Streamlit lo
#  interprete como bloque de código en vez de HTML)
# -------------------------------------------------------

def generar_pasos_html(pasos):
    items = []
    for i, paso in enumerate(pasos, 1):
        items.append(
            '<div class="step">'
            '<div class="step-number">' + f"{i:02d}" + '</div>'
            '<div>'
            '<div class="step-title">' + str(paso["titulo"]) + '</div>'
            '<div class="step-text">' + str(paso["texto"]) + '</div>'
            '</div>'
            '</div>'
        )
    return '<div class="step-container">' + ''.join(items) + '</div>'


def generar_resultado_html(nombre, cientifico, confianza, severidad, pasos, diagnostico_generado):
    """
    Genera TODO el cuadro cafecito claro (resultado + orientación) como
    un único bloque HTML, en un solo st.markdown(), para que el
    contenedor no se rompa y no se muestre código crudo.
    """
    encabezado = (
        '<div class="result-top">'
        '<div>'
        f'<div class="disease-title">{nombre}</div>'
        f'<div class="scientific">{cientifico}</div>'
        '</div>'
        '<div style="text-align:right">'
        f'<div class="metric-big">{confianza:.1f}%</div>'
        '<div class="metric-title">CONFIANZA</div>'
        '</div>'
        '</div>'
    )

    if diagnostico_generado:
        cuerpo = f'<div class="severity-pill">✅ Nivel de severidad detectado: {severidad}</div>'
        if pasos:
            cuerpo += (
                '<div class="orientacion-title">📋 Orientación y manejo preventivo</div>'
                '<div class="orientacion-sub">Aquí tienes una recomendación técnica detallada para manejar la situación:</div>'
                + generar_pasos_html(pasos)
            )
    else:
        cuerpo = '<div class="info-pill">ℹ️ Cargue una imagen para comenzar el diagnóstico.</div>'

    return '<div class="result-box">' + encabezado + cuerpo + '</div>'

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
    st.markdown('<div class="metric-title">ÚLTIMO DIAGNÓSTICO</div>', unsafe_allow_html=True)

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
    # MOSTRAR RESULTADO (un solo bloque HTML, todo dentro del cuadro)
    # ---------------------------------------

    resultado_html = generar_resultado_html(
        nombre, cientifico, confianza, severidad, pasos, diagnostico_generado
    )
    st.markdown(resultado_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------
    # HISTORIAL RECIENTE
    # ---------------------------------------

    st.markdown('<div class="metric-title" style="margin-top: 25px;">HISTORIAL RECIENTE</div>', unsafe_allow_html=True)

    if len(st.session_state.historial) == 0:
        st.markdown('<div class="history">No existen diagnósticos.</div>', unsafe_allow_html=True)
    else:
        from datetime import datetime
        fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M")
        for i, item in enumerate(st.session_state.historial):
            if i == 0:
                st.markdown(
                    '<div class="history">'
                    f'<strong>🟤 {item}</strong><br>'
                    f'<span style="font-size:12px; color:#8B8076;">{fecha_actual}</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f'<div class="history">🟤 {item}</div>', unsafe_allow_html=True)

    # ---------------------------------------
    # PIE DE PÁGINA
    # ---------------------------------------

    st.markdown("""
    <hr>
    <div class="footer">
        ☕ Servicio Web de Diagnóstico Foliar | UTH 2026
    </div>
    """, unsafe_allow_html=True)
