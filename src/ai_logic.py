import os
import json
import streamlit as st
from google import genai
from dotenv import load_dotenv

# Configuración Inicial de la api key google gemini
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = None

if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error conexión AI: {e}")

def extraer_filtros_con_ia(consulta_usuario):
    if not client: return {}
    prompt = f"""
    Eres un extractor de datos JSON. Analiza la frase: "{consulta_usuario}".
    Devuelve SOLAMENTE un objeto JSON válido (sin markdown).
    Campos: "color", "precio_max", "categoria", "marca".
    Ejemplo: "rojo barato" -> {{"color": "rojo", "precio_max": "bajo"}}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpio)
    except:
        return {}

def generar_respuesta_rag(consulta_usuario, productos, historial):
    if not client: return "Error: No hay conexión con la IA."
    # Validacion anti alucionacion: Si lista productos esta vacia, informa al usaurio y no inventa la IA
    if not productos:
        return "Lo siento, actualmente no tengo productos en el catálogo que coincidan con tu búsqueda. ¿Deseas intentar con otros términos?"

    # Contexto enriquecedor
    # Construimos el contexto usando 'rag_context' que ya incluye marca y opiniones reales
    contexto_prods = ""
    for p in productos:
        meta = p.get('metadata', {})
        
        # Extraemos el rag_context recomendado por el backend [cite: 45, 75]
        rag_info = meta.get('rag_context', 'Sin descripción detallada.')
        contexto_prods += f"- {meta.get('title', 'Producto')}: {rag_info}\n"
    contexto_chat = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in historial[-3:]])

    # Prompt de sistema estricto
    prompt = f"""
    Eres un asistente experto de Amazon. El sistema ya ha procesado la consulta del usuario
    (que pudo ser texto o imagen) y ha recuperado los siguientes productos relevantes.
    PRODUCTOS RECUPERADOS:
    {contexto_prods}

    HISTORIAL:
    {contexto_chat}
    USUARIO DICE: "{consulta_usuario}"

    INSTRUCCIONES:

    1. No menciones que no puedes ver imágenes. Actúa como si los PRODUCTOS RECUPERADOS fueran la respuesta directa a lo que el usuario necesita.
    2. Si el usuario subió una imagen, simplemente di: "Basado en el producto de tu imagen, te recomiendo..."
    3. Justifica usando los detalles técnicos y reseñas del contexto.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generando respuesta: {e}"