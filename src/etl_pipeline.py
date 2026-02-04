import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CSV_FILES = [
    "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv",
    "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
]

OUTPUT_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "images")
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed_products.csv")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def clean_asin(asin_raw):
    if pd.isna(asin_raw): return "UNKNOWN"
    return str(asin_raw).split(',')[0].strip()

def is_valid_url(url):
    url = str(url).lower()
    if "barcode" in url or "upccodesearch" in url or "barcodable" in url or "pixel" in url:
        return False
    return True

def clean_text(text):
    """Aplana el texto: quita saltos de linea y comillas conflictivas."""
    if pd.isna(text): return ""
    return str(text).replace('\n', ' ').replace('\r', '').replace('"', "'").strip()

def run_etl():
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    # 1. Carga de Datos
    dfs = []
    print("Iniciando carga de datasets...")
    for filename in CSV_FILES:
        filepath = os.path.join(PROJECT_ROOT, filename)
        if os.path.exists(filepath):
            try:
                # Leemos todo como string
                df_temp = pd.read_csv(filepath, dtype=str)
                dfs.append(df_temp)
            except Exception as e:
                print(f"   Error leyendo {filename}: {e}")

    if not dfs: return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Asegurar columnas clave
    for col in ['asins', 'imageURLs', 'name', 'primaryCategories', 'reviews.text', 'reviews.rating']:
        if col not in full_df.columns: full_df[col] = ""

    full_df['clean_id'] = full_df['asins'].apply(clean_asin)
    
    # 2. Pre-procesamiento de Reviews (Agregación Inteligente)
    print("Agrupando reseñas por producto...")
    
    # Creamos un diccionario donde la clave es el ID y el valor es una LISTA de reviews
    # Tomamos 'reviews.text' y 'reviews.rating' y los combinamos
    full_df['formatted_review'] = full_df.apply(
        lambda x: f"[{clean_text(x['reviews.rating'])}/5] {clean_text(x['reviews.text'])[:200]}", 
        axis=1
    )
    
    # Agrupar: Para cada ID, obtener una lista de todas sus reviews
    reviews_map = full_df.groupby('clean_id')['formatted_review'].apply(list).to_dict()

    # 3. Deduplicación de Productos
    print("🧹 Obteniendo productos únicos...")
    products = full_df.groupby('clean_id').first().reset_index()
    products = products[products['imageURLs'].notna() & (products['imageURLs'] != "nan")]
    
    print(f"Total productos únicos a procesar: {len(products)}")
    
    valid_products = []
    
    # 4. Descarga y Construcción Final
    for index, row in products.iterrows():
        raw_urls = str(row['imageURLs']).split(',')
        
        # Selección de imagen (lógica igual que antes)
        sorted_urls = []
        for u in raw_urls:
            u = u.strip()
            if is_valid_url(u) and len(u) > 10: 
                if "SL1500" in u: sorted_urls.insert(0, u)
                else: sorted_urls.append(u)
        
        if not sorted_urls: continue
            
        image_saved = False
        final_img_path = ""
        
        for url in sorted_urls:
            img_filename = f"{row['clean_id']}.jpg"
            img_path = os.path.join(OUTPUT_IMG_DIR, img_filename)
            
            if os.path.exists(img_path):
                image_saved = True
                final_img_path = img_path
                break
            
            try:
                response = requests.get(url, headers=HEADERS, timeout=4)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img.load()
                    if img.width < 100: continue 
                    img.convert('RGB').save(img_path)
                    image_saved = True
                    final_img_path = img_path
                    break
            except: continue
        
        if image_saved:
            brand = clean_text(row.get('brand', 'Unknown'))
            cat = clean_text(row.get('primaryCategories', ''))
            desc_text = f"{clean_text(row['name'])}. Category: {cat}. Brand: {brand}"
            
            # --- CONSTRUCCIÓN DEL CONTEXTO RAG MULTI-REVIEW ---
            # Recuperar lista de reviews de este producto
            prod_reviews = reviews_map.get(row['clean_id'], [])
            
            # Tomar máximo 3 reviews para no llenar el CSV ni el prompt
            top_reviews = prod_reviews[:3] 
            
            # Unirlas con un separador claro " || "
            reviews_str = " || ".join(top_reviews)
            
            rag_context = (
                f"Product: {clean_text(row['name'])} | "
                f"Brand: {brand} | "
                f"Reviews Summary: {reviews_str}"
            )

            relative_path = os.path.join("data", "images", f"{row['clean_id']}.jpg")
            
            valid_products.append({
                "id": row['clean_id'],
                "title": clean_text(row['name']),
                "category": cat,
                "brand": brand,
                "description": desc_text,
                "rag_context": rag_context, # Ahora contiene hasta 3 opiniones
                "image_path": relative_path
            })
            
            if len(valid_products) % 20 == 0:
                print(f"   {len(valid_products)} listos...")

    df_clean = pd.DataFrame(valid_products)
    df_clean.to_csv(PROCESSED_DATA, index=False)
    print(f"\nETL Finalizado. {len(df_clean)} productos guardados.")

if __name__ == "__main__":
    run_etl()