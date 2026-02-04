import streamlit as st
import os
from src.ai_logic import extraer_filtros_con_ia, generar_respuesta_rag
from src.retrieval import SearchEngine 

# Configuracion titulo pagina
st.set_page_config(page_title="E-commerce Multimodal", layout="wide")
st.title("🛍️ Asistente de Compras Inteligente")

# Cargar motor
@st.cache_resource
def load_engine():
    return SearchEngine()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error cargando el backend: {e}")
    st.stop()

# Gestion de estado
if "messages" not in st.session_state: st.session_state.messages = []
if "last_results" not in st.session_state: st.session_state.last_results = []
if "filtros" not in st.session_state: st.session_state.filtros = {} 

# Barra lateral
with st.sidebar:
    st.header("Configuración")
    uploaded_img = st.file_uploader("Imagen de referencia", type=['jpg', 'png', 'jpeg'])
    
    # --- INTERFAZ LIMPIA: Se eliminó el debug de memoria JSON ---
    
    if st.button("Limpiar Sesión"):
        st.session_state.clear()
        st.rerun()

# Mostrar historial del chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Logica principal
if prompt := st.chat_input("¿Qué buscas hoy?"):
    
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Analizar Intención y Actualizar Memoria
    nuevos_filtros = extraer_filtros_con_ia(prompt)
    
    # TRUCO DE MEMORIA: Solo actualizamos lo que sea nuevo, conservando lo viejo (ej. el producto "speaker")
    if nuevos_filtros:
        st.session_state.filtros.update(nuevos_filtros)
    
    # 2. Construir la Query Limpia basada en la Memoria Acumulada
    # Juntamos: Producto + Marca + Color + Categoria
    filtros_activos = st.session_state.filtros
    palabras_clave = [
        filtros_activos.get("producto"), # Esto es lo más importante (ej. "speaker")
        filtros_activos.get("marca"),
        filtros_activos.get("color"),
        filtros_activos.get("categoria")
    ]
    # Filtramos nulos y creamos el string de búsqueda
    query_limpia = " ".join([str(p) for p in palabras_clave if p])
    
    # Si la memoria está vacía (primer turno y falló extracción), usamos el prompt original
    if not query_limpia:
        query_limpia = prompt

    # Configuración de búsqueda
    query_para_backend = query_limpia
    umbral_corte = 0.15 
    tipo_busqueda = "Texto Inteligente"

    if uploaded_img:
        tipo_busqueda = "Imagen"
        umbral_corte = 0.60 
        with open("temp_query.jpg", "wb") as f:
            f.write(uploaded_img.getbuffer())
        query_para_backend = "temp_query.jpg"
        
        if len(prompt) > 5:
            st.warning("Nota: Priorizando imagen. Para buscar solo texto, elimina la imagen.")

    # Procesamiento
    st.write(f"🔍 Buscando: **'{query_para_backend}'** ({tipo_busqueda})...")
    
    try:
        resultados_crudos = engine.search(query=query_para_backend)
        resultados_filtrados = []

        if resultados_crudos:
            top_score = resultados_crudos[0].get('score', 0)
            st.caption(f"Debug: Score Top: {top_score:.4f} | Umbral: {umbral_corte}")

            # Filtro individual
            for producto in resultados_crudos:
                if producto.get('score', 0) >= umbral_corte:
                    resultados_filtrados.append(producto)

            # Filtro de Líder Solitario
            if len(resultados_filtrados) >= 2:
                diff = resultados_filtrados[0]['score'] - resultados_filtrados[1]['score']
                if diff > 0.10:
                    resultados_filtrados = [resultados_filtrados[0]]
            
            resultados = resultados_filtrados
            
            if not resultados:
                st.error("Resultados descartados por baja relevancia.")
        else:
            resultados = []

        st.session_state.last_results = resultados
        if resultados:
            st.success(f"¡{len(resultados)} encontrados!")
        else:
            st.info("No hay coincidencias.")

    except Exception as e:
        st.error(f"Error: {e}")

    # Generar Respuesta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta = generar_respuesta_rag(
                prompt, 
                st.session_state.last_results, 
                st.session_state.messages
            )
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

# Resultados Visuales
if st.session_state.last_results:
    st.divider()
    cols = st.columns(3)
    for i, item in enumerate(st.session_state.last_results):
        meta = item['metadata']
        with cols[i % 3]:
            try:
                st.image(meta['image_relative_path']) 
            except:
                st.warning("Sin imagen")
            st.caption(f"**{meta.get('title', 'Producto')}**")
            st.write(f"Rel: {item.get('score', 0):.2f}")