import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURACIÓN DE RUTAS ---
csv_folder = "Meta" 
descriptors_folder = "Datasets_Descriptores"
datasets_features = {
    "HOG": "dataset_hog.csv",
    "Hu": "dataset_momentos_hu.csv",
    "ORB": "dataset_orb.csv"
}

class_mapping = {
    "Mestizos.csv": 0,
    "Afro-Ecuadorians.csv": 1,
    "European_Descendants.csv": 2,
    "Indigenous.csv": 3
}

def clean_filename(name):
    base = os.path.splitext(str(name).strip().lower())[0]
    if "_processed" in base:
        base = base.replace("_processed", "")
    return base

# --- 2. DEFINICIÓN DE LA VARIABLE OBJETIVO (SUPER EXTRACTOR) ---
print("META 1 - PASO A: Cargando y definiendo variable objetivo (Clases)...")
image_labels = {}

for csv_name, class_id in class_mapping.items():
    csv_path = os.path.join(csv_folder, csv_name)
    if os.path.exists(csv_path):
        try:
            # Forzamos la lectura probando comas, puntos y comas, y diferentes encodados
            try:
                df_etnia = pd.read_csv(csv_path, encoding='utf-8')
            except:
                df_etnia = pd.read_csv(csv_path, encoding='latin1')
                
            if df_etnia.shape[1] <= 1:
                try:
                    df_etnia = pd.read_csv(csv_path, sep=';', encoding='utf-8')
                except:
                    df_etnia = pd.read_csv(csv_path, sep=';', encoding='latin1')
            
            # Limpiar nombres de columnas eliminando espacios o caracteres invisibles
            df_etnia.columns = df_etnia.columns.str.strip().str.lower()
            
            # Buscar cualquier columna que sirva como ID de archivo
            col_encontrada = None
            for col in df_etnia.columns:
                if 'filename' in col or 'sub_id' in col or 'file' in col or 'image' in col or 'id' in col:
                    col_encontrada = col
                    break
            
            # Si no encuentra ninguna por nombre, tomamos la primera columna por defecto
            if col_encontrada is None and df_etnia.shape[1] > 0:
                col_encontrada = df_etnia.columns[0]
                
            if col_encontrada is not None:
                for filename in df_etnia[col_encontrada].dropna().unique():
                    name_clean = clean_filename(filename)
                    if "_processed" in name_clean:
                        name_clean = name_clean.replace("_processed", "")
                    image_labels[name_clean] = class_id
            else:
                print(f"No se pudo encontrar ninguna columna útil en {csv_name}")
                
        except Exception as e:
            print(f"Error crítico al intentar abrir {csv_name}: {e}")
    else:
        print(f"Archivo no encontrado en la ruta: {csv_path}")

print(f"ℹTotal de imágenes mapeadas en las etnias: {len(image_labels)}")

# --- 3. VALIDACIÓN Y SPLIT PARA CADA DATASET ---
print("\nMETA 1 - PASO B & C: Validación de conjuntos y Split Entrenamiento/Prueba (80/20)...")

for name, csv_file in datasets_features.items():
    csv_path = os.path.join(descriptors_folder, csv_file)
    if not os.path.exists(csv_path):
        print(f"Saltando {name}: No se encontró el archivo {csv_path}")
        continue
        
    df_feat = pd.read_csv(csv_path)
    
    # Detectar dinámicamente como se llama la columna de identificacion en tu archivo de descriptores
    col_id = next((c for c in df_feat.columns if 'id' in c.lower() or 'file' in c.lower()), df_feat.columns[0])
    
    X_list = []
    y_list = []
    
    # Validacion y emparejamiento estricto por nombre base
    for _, row in df_feat.iterrows():
        raw_filename = row[col_id]
        cleaned_name = clean_filename(raw_filename)
        
        if cleaned_name in image_labels:
            features = row.drop(col_id).values.astype(np.float64)
            X_list.append(features)
            y_list.append(image_labels[cleaned_name])
            
    X_data = np.array(X_list)
    y_data = np.array(y_list)
    
    # Verificar si logramos recuperar muestras
    if len(X_data) == 0:
        print(f"Error: El dataset {name} no pudo emparejar ninguna muestra. Columna ID usada: '{col_id}'")
        print(f"   Ejemplo de ID en tu descriptor {name}: {list(df_feat[col_id].head(3))}")
        print(f"   Ejemplo de llaves esperadas de etnias: {list(image_labels.keys())[:3]}")
        continue
        
    # Limpieza automática de nulos por si acaso
    if np.isnan(X_data).any():
        X_data = np.nan_to_num(X_data)
        
    # Split Entrenamiento/Prueba (80% Train, 20% Test) con Estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    
    # Guardar las matrices procesadas
    np.save(os.path.join(descriptors_folder, f"X_train_{name}.npy"), X_train)
    np.save(os.path.join(descriptors_folder, f"X_test_{name}.npy"), X_test)
    np.save(os.path.join(descriptors_folder, f"y_train_{name}.npy"), y_train)
    np.save(os.path.join(descriptors_folder, f"y_test_{name}.npy"), y_test)
    
    # Imprimir reporte formal de evidencias para la Meta 1
    print("-" * 50)
    print(f"DATASET ENFOQUE: {name}")
    print(f"   🔹 Dimensión total de características: {X_data.shape[1]} columnas")
    print(f"   🔹 Validación: {len(X_data)} muestras limpias emparejadas con éxito.")
    print(f"   🔹 Split Entrenamiento (80%): {X_train.shape[0]} muestras")
    print(f"   🔹 Split Prueba (20%): {X_test.shape[0]} muestras")

print("\n¡Proceso finalizado!")