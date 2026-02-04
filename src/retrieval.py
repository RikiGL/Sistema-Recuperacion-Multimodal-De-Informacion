import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from PIL import Image
import os
import torch

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

# Modelos
EMBEDDING_MODEL = "clip-ViT-B-32"
# Modelo Cross-Encoder para el Re-ranking (más lento pero más preciso)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class SearchEngine:
    def __init__(self):
        """
        Carga los modelos y conecta a la BD una sola vez al iniciar la app.
        """
        print("Inicializando Motor de Búsqueda...")
        
        # 1. Detectar dispositivo (GPU/CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   -> Usando dispositivo: {self.device}")

        # 2. Cargar CLIP (para búsqueda rápida inicial)
        print("   -> Cargando modelo de Embeddings (CLIP)...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=self.device)

        # 3. Cargar Cross-Encoder (para re-ranking)
        print("   -> Cargando modelo de Re-ranking...")
        self.reranker = CrossEncoder(RERANKER_MODEL, device=self.device)

        # 4. Conectar a ChromaDB
        print(f"   -> Conectando a Base de Datos en: {DB_PATH}")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_collection("amazon_products")
        
        print("Motor listo.")

    def search(self, query, top_k_retrieval=20, top_k_final=5):
        """
        Realiza la búsqueda híbrida:
        1. Retrieval: Busca los 20 más parecidos con CLIP.
        2. Re-ranking: Ordena esos 20 usando el Cross-Encoder.
        """
        
        # --- PASO 1: RETRIEVAL (Búsqueda Vectorial) ---
        # Determinar si la query es texto o ruta de imagen
        is_image_query = False
        
        if os.path.exists(query) and query.lower().endswith(('.jpg', '.png', '.jpeg')):
            # Es una búsqueda IMAGEN-A-PRODUCTO
            print(f"Buscando por imagen: {query}")
            is_image_query = True
            image = Image.open(query)
            query_emb = self.embedder.encode(image).tolist()
            query_content = "Image Query" # Placeholder para el reranker visual si fuera necesario
        else:
            # Es una búsqueda TEXTO-A-PRODUCTO
            print(f"Buscando por texto: '{query}'")
            query_emb = self.embedder.encode(query).tolist()
            query_content = query

        # Consulta a ChromaDB
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k_retrieval,
            # Incluimos metadatos para mostrar info y documentos para el RAG
            include=['metadatas', 'distances'] 
        )

        # Formatear resultados iniciales
        candidates = []
        ids = results['ids'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0]

        for i in range(len(ids)):
            candidates.append({
                "id": ids[i],
                "score": 1 - distances[i], # Convertir distancia a similitud aprox
                "metadata": metas[i],
                "original_rank": i + 1
            })

        # --- PASO 2: RE-RANKING ---
        # El Cross-Encoder compara (Query, Documento) y da un score de relevancia real.
        # NOTA: Cross-Encoder funciona mejor Texto-Texto. 
        # Si la búsqueda es por IMAGEN, saltamos el re-ranking textual o usamos solo scores de CLIP.
        
        if not is_image_query:
            print("   -> Aplicando Re-ranking...")
            # Preparamos pares [Query, Texto del Producto]
            # Usamos la descripción completa del producto para comparar
            pairs = [[query, c['metadata']['description']] for c in candidates]
            
            # El modelo calcula scores para todos los pares
            rerank_scores = self.reranker.predict(pairs)
            
            # Actualizamos scores y ordenamos
            for i, cand in enumerate(candidates):
                cand['rerank_score'] = float(rerank_scores[i])
            
            # Ordenar descendente por el nuevo score del reranker
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        else:
            print("   -> Saltando Re-ranking (Búsqueda visual pura)")

        # --- RETORNO FINAL ---
        return candidates[:top_k_final]
'''
# Bloque de prueba rápida
if __name__ == "__main__":
    engine = SearchEngine()
    
    # Prueba Texto
    print("\n--- PRUEBA DE TEXTO ---")
    res = engine.search("rechargeable batteries", top_k_final=3)
    for r in res:
        print(f"[{r.get('rerank_score', r['score']):.4f}] {r['metadata']['title']}")
'''

if __name__ == "__main__":
    engine = SearchEngine()
    
    # 1. Elegimos una imagen que ya tengas descargada para probar
    # (Asegúrate de cambiar el nombre del archivo por uno que veas en tu carpeta data/images)
    test_image_path = os.path.join(PROJECT_ROOT, "data/test_samples/que_es_alexa_y_como_funciona_53622_orig.jpg")
    
    # Si no tienes esa exacta, busca cualquier .jpg en data/images y pon su nombre ahí.
    
    if os.path.exists(test_image_path):
        print(f"\n--- PRUEBA DE IMAGEN: {test_image_path} ---")
        
        # El motor detectará automáticamente que es una ruta de archivo .jpg
        results = engine.search(test_image_path, top_k_final=5)
        
        for i, r in enumerate(results):
            print(f"{i+1}. [Score: {r['score']:.4f}] {r['metadata']['title']}")
            print(f"   Archivo: {r['metadata']['image_relative_path']}")
    else:
        print(f"No encontré la imagen de prueba: {test_image_path}")
        print("Revisa la carpeta data/images y pon un nombre de archivo real.")