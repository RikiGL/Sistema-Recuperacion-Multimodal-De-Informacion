import os
import json
import streamlit as st
from google import genai
from dotenv import load_dotenv

# --- CONFIGURACIÓN DE MODELO ---
# Usamos Gemma 3 27B como pediste.
# Si te da error 404, prueba quitando el "-it" final, aunque el estándar es con "-it".
MODELO_ACTUAL = "gemma-3-27b-it"

# Configuración Inicial
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = None

if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error conexión AI: {e}")

def extraer_filtros_con_ia(consulta_usuario):
    """
    Extrae producto, color, categoría y marca usando Gemma 3.
    """
    if not client: return {}
    
    # Gemma necesita un prompt muy directo para JSON
    prompt = f"""
    Actúa como una API que convierte lenguaje natural a JSON.
    Analiza la consulta: "{consulta_usuario}"
    
    Tu objetivo es extraer:
    - "producto": El objeto que busca (tradúcelo a inglés si es posible, ej: "speaker", "keyboard").
    - "marca": La marca (ej: Sony, Dell).
    - "color": El color.
    - "categoria": La categoría general.

    REGLAS ESTRICTAS:
    1. Responde ÚNICAMENTE con el objeto JSON.
    2. No añadas texto introductorio ni explicaciones.
    3. Si un campo no se menciona, omítelo.

    Ejemplo Entrada: "Quiero un teclado gamer logitech"
    Ejemplo Salida: {{"producto": "gaming keyboard", "marca": "Logitech"}}
    """
    
    try:
        response = client.models.generate_content(
            model=MODELO_ACTUAL, 
            contents=prompt
        )
        
        # Limpieza agresiva porque Gemma a veces es conversacional
        texto_limpio = response.text
        texto_limpio = texto_limpio.replace("```json", "").replace("```", "").strip()
        
        # Truco: Si Gemma responde algo como "Aquí está el JSON: { ... }", nos quedamos solo con lo que está entre llaves
        if "{" in texto_limpio: 
            texto_limpio = "{" + texto_limpio.split("{", 1)[1]
        if "}" in texto_limpio:
            texto_limpio = texto_limpio.rsplit("}", 1)[0] + "}"
            
        diccionario = json.loads(texto_limpio)
        # Limpiamos valores nulos
        return {k: v for k, v in diccionario.items() if v}
    except:
        return {}

def generar_respuesta_rag(consulta_usuario, productos, historial):
    if not client: return "Error: No hay conexión con la IA."
    
    if not productos:
        return "Lo siento, no encontré productos que coincidan exactamente. Intenta con términos más generales."

    # Contexto enriquecedor
    contexto_prods = ""
    for p in productos:
        meta = p.get('metadata', {})
        rag_info = meta.get('rag_context', 'Sin descripción.')
        # Incluimos el score para que la IA sepa cuál es más relevante
        contexto_prods += f"- {meta.get('title', 'Producto')} (Relevancia: {p.get('score', 0):.2f}): {rag_info}\n"
    
    # Historial corto
    contexto_chat = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in historial[-3:]])

    prompt = f"""
    Eres un asistente de ventas experto.
    
    PRODUCTOS DISPONIBLES (Ordenados por relevancia):
    {contexto_prods}

    CONTEXTO DE LA CONVERSACIÓN:
    {contexto_chat}
    
    CLIENTE PREGUNTA: "{consulta_usuario}"

    INSTRUCCIONES:
    1. Recomienda el producto más relevante de la lista anterior.
    2. Si el cliente pidió una característica específica (ej. "rojo") y uno de los productos la tiene, menciónalo.
    3. Sé breve y profesional. No inventes productos que no estén en la lista.
    """
    try:
        response = client.models.generate_content(
            model=MODELO_ACTUAL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        # Fallback de seguridad: Si Gemma 3 falla, intentamos con Flash Lite que tenías libre
        try:
            print(f"Fallo Gemma 3 ({e}), usando backup...")
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite-preview-02-05",
                contents=prompt
            )
            return response.text
        except:
            return f"Error generando respuesta ({MODELO_ACTUAL}): {e}"