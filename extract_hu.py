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

print("🚀 Extrayendo Momentos de Hu invariantivos...")
start_total_time = time.time()

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        try:
            # Lectura directa de la máscara binarizada/procesada de la silueta
            thresh = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if thresh is None: continue
            
            moments = cv2.moments(thresh)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Escala logarítmica para estabilizar magnitudes dinámicas dispares
            hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            clean_id = os.path.splitext(filename)[0]
            hu_row = {"sub_id": clean_id}
            for i, val in enumerate(hu_log):
                hu_row[f"hu_{i+1}"] = val
            dataset_hu.append(hu_row)
            
        except Exception as e:
            print(f"❌ Error en {filename}: {e}")

end_total_time = time.time()
total_execution_time = end_total_time - start_total_time

df_hu = pd.DataFrame(dataset_hu)
output_file_name = "dataset_momentos_hu.csv"
df_hu.to_csv(os.path.join(output_folder, output_file_name), index=False)

print("\n" + "="*50)
print("📊 REPORTE TÉCNICO COMPLETO - DESCRIPTOR: MOMENTOS DE HU")
print("="*50)
print(f"🔹 Dimensiones del dataset: {df_hu.shape[0]} muestras x {df_hu.shape[1] - 1} características")
print(f"🔹 Tiempo total: {total_execution_time:.4f} segundos")
print("="*50 + "\n")