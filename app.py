import streamlit as st
import os
from src.ai_logic import extraer_filtros_con_ia, generar_respuesta_rag
from src.retrieval import SearchEngine 

st.set_page_config(page_title="E-commerce Multimodal", layout="wide")
st.title("Asistente de Compras Inteligente")

# Carga del motor con cache 
@st.cache_resource
def load_engine():
    return SearchEngine()

try:
    engine = load_engine()
    # Inidcador de motor de busqueda conectado (opcional)
except Exception as e:
    st.error(f"Error cargando el backend: {e}")
    st.stop()

# Gestion de estados
if "messages" not in st.session_state: st.session_state.messages = []
if "last_results" not in st.session_state: st.session_state.last_results = []
if "filtros" not in st.session_state: st.session_state.filtros = {} 

# Barra lateral
with st.sidebar:
    st.header("Configuración")
    uploaded_img = st.file_uploader("Imagen de referencia", type=['jpg', 'png', 'jpeg'])
    
    with st.expander("Debug Memoria"):
        st.json(st.session_state.filtros)
    
    if st.button("Limpiar Sesión"):
        st.session_state.clear()
        st.rerun()

# Mostrar chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Logica princial
if prompt := st.chat_input("¿Qué buscas hoy?"):
    # Guardar y mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Definir la consulta (Texto o Imagen)
    query_para_backend = prompt
    
    if uploaded_img:
        # Guardamos la imagen temporalmente como pide el backend
        with open("temp_query.jpg", "wb") as f:
            f.write(uploaded_img.getbuffer())
        query_para_backend = "temp_query.jpg"

    # Procesamiento (Status y Busqueda)
    with st.status("Procesando...", expanded=False) as status:
        st.write("Entendiendo intención...")
        nuevos_filtros = extraer_filtros_con_ia(prompt)
        st.session_state.filtros.update(nuevos_filtros)
        
        st.write(f"Consultando motor con: {query_para_backend}...")
        try:
            resultados = engine.search(query=query_para_backend)
            st.session_state.last_results = resultados
            status.update(label="¡Resultados encontrados!", state="complete")
        except Exception as e:
            st.error(f"Fallo en búsqueda: {e}")
            status.update(label="Error", state="error")

    # Generar Respuesta RAG  
    with st.chat_message("assistant"):
        with st.spinner("Analizando productos..."):
            respuesta = generar_respuesta_rag(
                prompt, 
                st.session_state.last_results, 
                st.session_state.messages
            )
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

# Resultados visuales
if st.session_state.last_results:
    st.divider()
    st.subheader("Productos Encontrados: ")
    cols = st.columns(3)
    for i, item in enumerate(st.session_state.last_results):
        meta = item['metadata']
        with cols[i % 3]:
            try:
                st.image(meta['image_relative_path']) 
            except:
                st.warning("Imagen no encontrada")
            
            st.caption(f"**{meta.get('title', 'Producto')}**")
            st.write(f"Relevancia: {item.get('score', 0):.2f}")