import streamlit as st
# Importamos nuestros módulos nuevos
from src.ai_logic import extraer_filtros_con_ia, generar_respuesta_rag
from src.retrieval import mock_search_engine

st.set_page_config(page_title="E-commerce Multimodal", layout="wide")
st.title(" Asistente de Compras Inteligente")

# 1. GESTIÓN DE ESTADO (Memoria)
if "messages" not in st.session_state: st.session_state.messages = []
if "last_results" not in st.session_state: st.session_state.last_results = []
if "filtros" not in st.session_state: st.session_state.filtros = {} 

# 2. BARRA LATERAL
with st.sidebar:
    st.header("Configuración")
    uploaded_img = st.file_uploader("Imagen de referencia", type=['jpg', 'png'])
    
    with st.expander("🔧 Debug Memoria"):
        st.json(st.session_state.filtros)
    
    if st.button("Limpiar Sesión"):
        st.session_state.clear()
        st.rerun()

# 3. MOSTRAR CHAT
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. LÓGICA PRINCIPAL
if prompt := st.chat_input("¿Qué buscas hoy?"):
    # A. Guardar input usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # B. Procesamiento (NLU + Búsqueda)
    with st.status("Procesando...", expanded=False) as status:
        # Extraer filtros (Usamos ai_logic.py)
        nuevos_filtros = extraer_filtros_con_ia(prompt)
        st.session_state.filtros.update(nuevos_filtros)
        
        # Buscar productos (Usamos retrieval.py)
        resultados = mock_search_engine(prompt, uploaded_img, st.session_state.filtros)
        st.session_state.last_results = resultados
        
        status.update(label="¡Completado!", state="complete")

    # C. Generar Respuesta (RAG)
    with st.chat_message("assistant"):
        with st.spinner("Escribiendo..."):
            respuesta = generar_respuesta_rag(
                prompt, 
                resultados, 
                st.session_state.messages
            )
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

# 5. RESULTADOS VISUALES
if st.session_state.last_results:
    st.divider()
    cols = st.columns(len(st.session_state.last_results))
    for i, p in enumerate(st.session_state.last_results):
        with cols[i]:
            st.image(p["img_url"])
            st.caption(f"**{p['title']}**\n{p['price']}")