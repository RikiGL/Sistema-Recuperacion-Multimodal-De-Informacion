# src/ai_logic.py
import os
import json
import streamlit as st
from google import genai
from dotenv import load_dotenv

# Configuración Inicial (Se ejecuta al importar)
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
            model='gemini-2.5-flash', # Modelo rápido
            contents=prompt
        )
        texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpio)
    except:
        return {}

def generar_respuesta_rag(consulta_usuario, productos, historial):
    if not client: return "Error: No hay conexión con la IA."

    contexto_prods = "\n".join([f"- {p['title']} ({p['price']}): {p['desc']}" for p in productos])
    contexto_chat = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in historial[-3:]])

    prompt = f"""
    Eres un experto en ventas. Recomienda basándote SOLO en:
    {contexto_prods}
    
    Historial:
    {contexto_chat}
    
    Usuario: "{consulta_usuario}"
    
    Instrucciones: Recomienda el mejor producto y justifica con sus características.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generando respuesta: {e}"