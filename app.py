import streamlit as st
import os
from src.ai_logic import extraer_filtros_con_ia, generar_respuesta_rag
from src.retrieval import SearchEngine 

# Configuracion titulo pagina
st.set_page_config(page_title="E-commerce Multimodal", layout="wide")
st.title("Asistente de Compras Inteligente")

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
    
    with st.expander("Debug Memoria"):
        st.json(st.session_state.filtros)
    
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

    # Configuración Dinámica de Búsqueda (HÍBRIDO)
    query_para_backend = prompt
    umbral_corte = 0.15  # Umbral bajo para TEXTO (CLIP es estricto texto-imagen)
    tipo_busqueda = "Texto"

    if uploaded_img:
        tipo_busqueda = "Imagen"
        umbral_corte = 0.60  # Umbral ALTO para evitar alucinaciones
        
        # Guardamos la imagen temporalmente
        with open("temp_query.jpg", "wb") as f:
            f.write(uploaded_img.getbuffer())
        query_para_backend = "temp_query.jpg"

        # Advertencia de usabilidad
        if len(prompt) > 5:
            st.warning("Nota: Se está priorizando la búsqueda por IMAGEN. Si quieres buscar solo por texto, elimina la imagen de la barra lateral.")

    # Procesamiento y Búsqueda
    st.write(f"Procesando solicitud por **{tipo_busqueda}**...")
    
    # 1. Extraer filtros (NLU)
    nuevos_filtros = extraer_filtros_con_ia(prompt)
    st.session_state.filtros.update(nuevos_filtros)
    
    # Buscar en Backend
    try:
        resultados_crudos = engine.search(query=query_para_backend)
        resultados_filtrados = []

        # Filtro individual
        if resultados_crudos:
            # Mostrar el puntaje del mejor para tu referencia 
            top_score = resultados_crudos[0].get('score', 0)
            st.caption(f"🔍 Debug: Similitud Top: **{top_score:.4f}** | Umbral: **{umbral_corte}**")

            # Bucle de Limpieza: Solo pasan los que superan el umbral
            for producto in resultados_crudos:
                score_prod = producto.get('score', 0)
                if score_prod >= umbral_corte:
                    resultados_filtrados.append(producto)

            # Si el primero es mucho mejor que el segundo (diferencia de 0.10), nos quedamos solo con el primero
            if len(resultados_filtrados) >= 2:
                diferencia = resultados_filtrados[0]['score'] - resultados_filtrados[1]['score']
                if diferencia > 0.10:
                    resultados_filtrados = [resultados_filtrados[0]]
                    st.caption("Se filtraron resultados secundarios por baja relevancia comparativa.")

            # Actualizamos la variable final
            resultados = resultados_filtrados

            # 4Verificar si quedó alguien vivo
            if not resultados:
                st.error(f"Resultados descartados. El mejor score ({top_score:.2f}) no fue suficiente o era confuso.")
        else:
            resultados = []

        st.session_state.last_results = resultados
        if resultados:
            st.success(f"¡{len(resultados)} resultados relevantes encontrados!")
        else:
            st.info("No se encontraron coincidencias suficientes en el catálogo.")

    except Exception as e:
        st.error(f"Fallo en búsqueda: {e}")

    # Generar Respuesta (RAG)
    with st.chat_message("assistant"):
        with st.spinner("Analizando productos..."):
            respuesta = generar_respuesta_rag(
                prompt, 
                st.session_state.last_results, 
                st.session_state.messages
            )
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

# RESULTADOS VISUALES 
if st.session_state.last_results:
    st.divider()
    st.subheader("🔍 Productos Encontrados")
    cols = st.columns(3)
    for i, item in enumerate(st.session_state.last_results):
        meta = item['metadata']
        with cols[i % 3]:
            try:
                # Mostramos la imagen del producto
                st.image(meta['image_relative_path']) 
            except:
                st.warning("Imagen no disponible")
            
            st.caption(f"**{meta.get('title', 'Producto')}**")
            st.write(f"Relevancia: {item.get('score', 0):.2f}")