# 🌿 - Detección de Enfermedades en Café

<div align="center">
  <img width="1200" alt="AgroDetect Banner" src="https://github.com/user-attachments/assets/6f82a43b-00d7-4632-9061-21b9c4479239" />
</div>


Sistema web interactivo basado en Deep Learning y Computación en la Nube para el diagnóstico automático de enfermedades foliares en el cultivo de café (*Coffea arabica* / *Coffea canephora*) y generación de informes agrónomos con Inteligencia Artificial Generativa.

---

## 📐 Arquitectura del Sistema

El sistema utiliza una arquitectura cliente-servidor desacoplada basada en la nube:

1. **Interfaz y Procesamiento (Streamlit Cloud):** Captura/carga de imagen, preprocesamiento tensorial e inferencia local con la red neuronal.
2. **Motor de Visión por Computadora:** Modelo MobileNetV2 mediante *Transfer Learning* entrenado en Keras/TensorFlow.
3. **Generación de Diagnóstico (Groq Cloud API):** Asesoría agrónoma generada por el LLM `llama-3.1-8b-instant`.

---

## ☁️ Servicios en la Nube Utilizados

* **Google Colab:** Entorno GPU para la reestructuración, data augmentation y entrenamiento del modelo visual.
* **GitHub:** Control de versiones y alojamiento del código fuente.
* **Streamlit Community Cloud:** Despliegue en la nube (*PaaS*) de la aplicación web.
* **Groq Cloud API:** Inferencia ultrafast para modelos de lenguaje (*LLMaaS*).

---

## 🛠️ Tecnologías Empleadas

* **Lenguaje:** Python 3.10+
* **Deep Learning:** TensorFlow 2.x, Keras, MobileNetV2
* **Procesamiento de Imágenes:** Pillow (PIL), NumPy
* **Desarrollo Web:** Streamlit
* **IA Generativa:** Groq SDK / OpenAI API (Model: `llama-3.1-8b-instant`)

---

## 🔄 Flujo de Funcionamiento

1. **Entrada de Datos:** El usuario captura una foto con la cámara o sube un archivo (`.jpg`, `.png`).
2. **Inferencia Visual:** La imagen se reescala a 224x224, se normaliza con `preprocess_input` y pasa por el modelo `agrodetect_mobilenetv2.keras`.
3. **Clasificación:** Se evalúan las 5 clases (*cercospora*, *minador*, *phoma*, *roya*, *sana*) y se determina la clase con mayor probabilidad.
4. **Informe Agrónomo:** Se envía el diagnóstico y porcentaje de confianza a la API de Groq, que devuelve recomendaciones de manejo preventivo y correctivo en tiempo real.

---

## ⚙️ Instrucciones de Instalación y Ejecución Local

### Prerrequisitos
* Python 3.9 o superior instalado.
* Clave API de Groq (`GROQ_API_KEY`).

### Pasos de Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/fabriciobeltrand25/Cafe_detectar_efermeda.git]
   cd cafe_detectar_efermeda
