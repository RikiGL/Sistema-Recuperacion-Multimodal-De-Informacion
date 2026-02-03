
# Multimodal E-Commerce Search Engine with RAG

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AI](https://img.shields.io/badge/GenAI-Gemini-orange)
![Vector DB](https://img.shields.io/badge/VectorDB-Chroma-green)
![Framework](https://img.shields.io/badge/Frontend-Streamlit-red)

Un sistema avanzado de recuperación de información diseñado para e-commerce que integra búsqueda semántica, capacidades multimodales (texto e imagen) y Generación Aumentada por Recuperación (RAG).

El objetivo de este proyecto es resolver las limitaciones de los buscadores tradicionales mediante la implementación de embeddings compartidos (CLIP), re-ranking neural para alta precisión y un asistente conversacional capaz de justificar recomendaciones basándose en evidencia real del producto.

## ⚡ Características Principales

* **Búsqueda Multimodal (Text-to-Product & Image-to-Product):** Permite a los usuarios buscar productos describiéndolos en lenguaje natural o subiendo una imagen de referencia, utilizando modelos **CLIP** para alinear ambos espacios vectoriales.
* **Pipeline de Re-ranking:** Implementación de una arquitectura de dos etapas:
    1.  *Retrieval:* Búsqueda rápida de candidatos top-k mediante similitud de coseno en **ChromaDB**.
    2.  *Re-ranking:* Refinamiento de precisión utilizando **Cross-Encoders** para reordenar los resultados según su relevancia semántica profunda.
* **Asistente RAG Contextual:** Un agente conversacional impulsado por **Google Gemini** que analiza los metadatos y reseñas de los productos recuperados para generar respuestas fundamentadas, evitando alucinaciones.
* **Memoria de Sesión:** Gestión de estado para permitir refinamiento iterativo de búsquedas (e.g., "muéstrame opciones más baratas" o "cambia el color").

## 🛠️ Arquitectura del Proyecto

El sistema sigue una arquitectura modular desacoplada:

```text
├── data/                      # Persistencia de datos
│   ├── chroma_db/             # Vector Store (ChromaDB)
│   ├── images/                # Repositorio local de imágenes de productos
│   └── processed_products.csv # Dataset normalizado con metadatos enriquecidos
├── src/                       # Core Logic
│   ├── etl_pipeline.py        # Pipeline de ingestión, limpieza y descarga de assets
│   ├── processing.py          # Generación de embeddings (CLIP) e indexación
│   ├── retrieval.py           # Motor de búsqueda híbrido (Search Engine + Reranker)
│   └── ai_logic.py            # Orquestación de LLM (Gemini) y extracción de entidades
├── app.py                     # Interfaz de usuario interactiva (Streamlit)
└── requirements.txt           # Dependencias del entorno

```

## 🚀 Instalación y Despliegue

### 1. Clonar el repositorio

```bash
git clone https://github.com/RikiGL/Sistema-Recuperacion-Multimodal-De-Informacion.git
cd Sistema-Recuperacion-Multimodal-De-Informacion

```

### 2. Configuración del Entorno

Se recomienda usar un entorno virtual. Crea un archivo `.env` en la raíz con tu API Key de Gemini:

```env
GEMINI_API_KEY="tu_api_key_aqui"

```

Instala las dependencias:

```bash
pip install -r requirements.txt

```

### 3. Ingesta de Datos (ETL)

El sistema utiliza el dataset [*Consumer Reviews of Amazon Products*](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data). Para maximizar el volumen de datos, el pipeline está diseñado para fusionar múltiples fuentes.

1. Descarga los siguientes dos archivos desde Kaggle:
* `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv`
* `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`


2. Colócalos en la raíz del proyecto.
3. Ejecuta el pipeline:

```bash
python -m src.etl_pipeline

```

*Este proceso concatenará ambos archivos, limpiará los datos, descargará las imágenes de los productos y generará el archivo unificado `processed_products.csv`.*

### 4. Indexación Vectorial

Genera los embeddings y puebla la base de datos vectorial ChromaDB:

```bash
python -m src.processing

```

### 5. Ejecución

Lanza la aplicación web:

```bash
streamlit run app.py

```

---

## 👨‍💻 Equipo y Contribuciones

Este proyecto fue desarrollado colaborativamente con una clara separación de responsabilidades en la arquitectura full-stack de IA.

### **Kevin Martinez ([@Al3xMR](https://github.com/Al3xMR))**

**Backend Engineer & Data Architect**

* Diseño e implementación del pipeline ETL (`etl_pipeline.py`) para la unificación y limpieza de múltiples datasets CSV, así como la gestión automatizada de assets multimedia.
* Arquitectura del sistema de indexación vectorial (`processing.py`) e integración con ChromaDB.
* Desarrollo del núcleo del motor de búsqueda (`retrieval.py`), implementando la lógica de recuperación multimodal y optimización mediante Cross-Encoders (Re-ranking).

### **Riki Guallichico ([@RikiGL](https://github.com/RikiGL))**

**Frontend Developer & AI Engineer**

* Desarrollo de la interfaz de usuario interactiva y gestión de estado en Streamlit (`app.py`).
* Ingeniería de Prompts e integración de Modelos de Lenguaje (`ai_logic.py`) para la funcionalidad RAG.
* Implementación de la lógica de memoria conversacional y extracción de filtros mediante procesamiento de lenguaje natural.

