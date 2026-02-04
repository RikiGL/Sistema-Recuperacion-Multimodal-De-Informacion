import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import torch

# --- CONFIGURACIÓN DE RUTAS ---
# Ubicación de este script (src/processing.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Raíz del proyecto (donde está data/ y el archivo .env si usaras)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed_products.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

# Nombre de la colección en ChromaDB
COLLECTION_NAME = "amazon_products"

# Modelo Multimodal (CLIP)
# Usamos uno ligero y eficiente para CPU: CLIP ViT-B-32
MODEL_NAME = "clip-ViT-B-32"

def process_and_index():
    print(f"Iniciando proceso de indexación...")
    
    # 1. Cargar el CSV limpio
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: No se encuentra {CSV_PATH}. Ejecuta el ETL primero.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Dataset cargado: {len(df)} productos.")

    # 2. Inicializar Modelo de Embeddings (Sentence-Transformers)
    print(f"Cargando modelo {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   -> Usando dispositivo: {device}")
    
    # Este modelo convierte IMÁGENES y TEXTO al mismo espacio vectorial
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 3. Inicializar ChromaDB (Base de datos vectorial persistente)
    print(f"Conectando a ChromaDB en: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Borrar colección anterior si existe (para empezar limpio)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("   -> Colección anterior eliminada.")
    except Exception:
        pass # Si no existe, no pasa nada, seguimos adelante.
        
    # Crear colección. Usamos cosine similarity space
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} 
    )

    # 4. Generar Embeddings e Insertar
    print("⚡ Generando embeddings (esto puede tardar unos minutos)...")
    
    ids = []
    embeddings = []
    metadatas = []
    
    successful_count = 0
    
    for index, row in df.iterrows():
        try:
            # Reconstruir ruta absoluta de la imagen
            # CSV tiene: "data/images/foto.jpg" -> OS necesita: "C:/.../data/images/foto.jpg"
            relative_path = row['image_path']
            full_image_path = os.path.join(PROJECT_ROOT, relative_path)
            
            if not os.path.exists(full_image_path):
                print(f"   Imagen no encontrada, saltando: {full_image_path}")
                continue
                
            # --- MAGIA MULTIMODAL ---
            # Indexamos la IMAGEN como el vector principal.
            # Como CLIP alinea texto e imagen, luego podremos buscar usando texto 
            # y encontrará esta imagen.
            image = Image.open(full_image_path)
            
            # Generar vector (list of floats)
            vector = model.encode(image).tolist()
            
            # Preparar metadatos para recuperarlos luego en la UI
            meta = {
                "product_id": str(row['id']),
                "title": str(row['title']),
                "category": str(row['category']),
                "brand": str(row['brand']),
                "description": str(row['description']),
                "rag_context": str(row['rag_context']),
                "image_relative_path": relative_path # Guardamos ruta relativa para la UI
            }
            
            ids.append(str(row['id']))
            embeddings.append(vector)
            metadatas.append(meta)
            
            successful_count += 1
            
            # Imprimir progreso
            if successful_count % 50 == 0:
                print(f"   {successful_count} productos procesados...")
                
        except Exception as e:
            print(f"   Error procesando ID {row.get('id', 'unknown')}: {e}")

    # 5. Guardar en lote (Batch upsert) en ChromaDB
    if ids:
        print(f"Insertando {len(ids)} vectores en la base de datos...")
        # Chroma tiene un límite de lote por defecto, a veces conviene partirlo si son miles
        # Para <10,000 suele aguantar de una, pero por seguridad lo hacemos en lotes pequeños si falla
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print("¡Indexación completada con éxito!")
        print(f"   Total indexado: {collection.count()} documentos.")
    else:
        print("No se generaron embeddings válidos.")

if __name__ == "__main__":
    process_and_index()