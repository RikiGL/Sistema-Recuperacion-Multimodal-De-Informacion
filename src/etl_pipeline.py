import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Lista de archivos a combinar
CSV_FILES = [
    "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
    "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
]

OUTPUT_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "images")
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed_products.csv")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def clean_asin(asin_raw):
    """Limpia el ASIN. Si vienen varios 'B001,B002', toma el primero."""
    if pd.isna(asin_raw): return "UNKNOWN"
    return str(asin_raw).split(',')[0].strip()

def is_valid_url(url):
    """Filtra URLs basura."""
    url = str(url).lower()
    if "barcode" in url or "upccodesearch" in url or "barcodable" in url or "pixel" in url:
        return False
    return True

def run_etl():
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    # 1. Fusión de CSVs
    dfs = []
    print("🔄 Iniciando carga y fusión de datasets...")
    
    for filename in CSV_FILES:
        filepath = os.path.join(PROJECT_ROOT, filename)
        if os.path.exists(filepath):
            print(f"   -> Leyendo: {filename}")
            try:
                # Leemos todo como string para evitar errores de tipo
                df_temp = pd.read_csv(filepath, dtype=str)
                dfs.append(df_temp)
            except Exception as e:
                print(f"   ⚠️ Error leyendo {filename}: {e}")
        else:
            print(f"   ❌ Archivo no encontrado: {filename}")

    if not dfs:
        print("❌ No se encontraron datos para procesar.")
        return

    # Concatenar todos los dataframes
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Total de filas crudas combinadas: {len(full_df)}")

    # 2. Limpieza y Deduplicación
    print("🧹 Limpiando y deduplicando...")
    
    # Asegurar que existan las columnas clave
    required_cols = ['asins', 'imageURLs', 'name', 'primaryCategories']
    for col in required_cols:
        if col not in full_df.columns:
            full_df[col] = ""

    full_df['clean_id'] = full_df['asins'].apply(clean_asin)
    
    # Agrupar por ID único. .first() toma el primer valor válido encontrado
    products = full_df.groupby('clean_id').first().reset_index()
    
    # Eliminar productos sin URL de imagen
    products = products[products['imageURLs'].notna() & (products['imageURLs'] != "nan")]
    
    # --- AQUÍ QUITAMOS EL LÍMITE ---
    # Antes había: products = products.head(MAX_PRODUCTS)
    # Ahora procesamos TODO lo que haya quedado tras el filtro.
    
    print(f"📦 Total de Productos Únicos detectados para descargar: {len(products)}")
    
    valid_products = []
    
    # 3. Descarga de Imágenes
    print("⬇️ Iniciando descarga masiva de imágenes...")
    
    for index, row in products.iterrows():
        raw_urls = str(row['imageURLs']).split(',')
        
        # Priorizar imágenes grandes
        sorted_urls = []
        for u in raw_urls:
            u = u.strip()
            if is_valid_url(u) and len(u) > 10: 
                if "SL1500" in u:
                    sorted_urls.insert(0, u)
                else:
                    sorted_urls.append(u)
        
        if not sorted_urls: continue
            
        image_saved = False
        final_img_path = ""
        
        for url in sorted_urls:
            img_filename = f"{row['clean_id']}.jpg"
            img_path = os.path.join(OUTPUT_IMG_DIR, img_filename)
            
            # Si ya existe, la reutilizamos
            if os.path.exists(img_path):
                image_saved = True
                final_img_path = img_path
                break
                
            try:
                # Timeout un poco más generoso por si el server es lento
                response = requests.get(url, headers=HEADERS, timeout=5)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img.load()
                    if img.width < 100: continue 
                        
                    img_rgb = img.convert('RGB')
                    img_rgb.save(img_path)
                    image_saved = True
                    final_img_path = img_path
                    break
            except Exception:
                continue
        
        if image_saved:
            brand = str(row.get('brand', ''))
            desc_text = f"{row['name']}. Category: {row['primaryCategories']}. Brand: {brand}"
            
            # Calculamos la ruta relativa al proyecto (data/images/foto.jpg)
            # Esto funcionará en Windows, Linux y Mac
            relative_path = os.path.join("data", "images", f"{row['clean_id']}.jpg")

            valid_products.append({
                "id": row['clean_id'],
                "title": row['name'],
                "category": row['primaryCategories'],
                "brand": brand,
                "description": desc_text,
                "image_path": relative_path
            })
            
            # Imprimir progreso cada 10 para ver que avanza
            if len(valid_products) % 10 == 0:
                print(f"   ✅ {len(valid_products)} productos listos...")

    # Guardar CSV final
    df_clean = pd.DataFrame(valid_products)
    df_clean.to_csv(PROCESSED_DATA, index=False)
    print(f"\n🎉 ETL Finalizado.")
    print(f"Total final de productos válidos: {len(df_clean)}")
    print(f"Archivo guardado en: {PROCESSED_DATA}")

if __name__ == "__main__":
    run_etl()