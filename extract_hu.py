import cv2
import os
import numpy as np
import pandas as pd
import time

input_folder = "Etnias_Procesadas" 
output_folder = "Datasets_Descriptores"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

dataset_hu = []

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

print("Extrayendo Momentos de Hu...")
start_total_time = time.time()

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        try:
            thresh = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if thresh is None: continue
            
            moments = cv2.moments(thresh)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            hu_row = {"sub_id": filename}
            for i, val in enumerate(hu_log):
                hu_row[f"hu_{i+1}"] = val
            dataset_hu.append(hu_row)
            
        except Exception as e:
            print(f"Error en {filename}: {e}")

end_total_time = time.time()
total_execution_time = end_total_time - start_total_time

df_hu = pd.DataFrame(dataset_hu)
output_file_name = "dataset_momentos_hu.csv"
df_hu.to_csv(os.path.join(output_folder, output_file_name), index=False)

# --- IMPRESIÓN DE MÉTRICAS COMPARATIVAS ---
print("\n" + "="*50)
print("COMPARACIÓN TÉCNICA - DESCRIPTOR: MOMENTOS DE HU")
print("="*50)
print(f"Número de características: {df_hu.shape[1] - 1} columnas (más la columna 'sub_id')")
print(f"Formato de salida: {os.path.splitext(output_file_name)[1].upper()} ({output_file_name})")
print(f"Tiempo de extracción total: {total_execution_time:.4f} segundos")
print(f"Costo computacional promedio: {(total_execution_time / max(1, len(df_hu))) * 1000:.2f} ms por imagen")
print(f"Representación de datos: Valores numéricos continuos (Punto flotante / Float64, tras escala logarítmica)")
print("="*50 + "\n")