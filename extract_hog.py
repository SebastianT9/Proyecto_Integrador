import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import hog
import time

input_folder = "Etnias" 
output_folder = "Datasets_Descriptores"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

dataset_hog = []

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

print("🚀 Extrayendo vectores HOG...")
start_total_time = time.time()

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        try:
            image = cv2.imread(input_path)
            if image is None: continue
            
            image_resized = cv2.resize(image, (128, 128))
            denoised = cv2.bilateralFilter(image_resized, d=9, sigmaColor=75, sigmaSpace=75)
            enhanced = adjust_gamma(denoised, gamma=1.5)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            hog_features = hog(
                gray, 
                orientations=9, 
                pixels_per_cell=(16, 16), 
                cells_per_block=(2, 2), 
                visualize=False
            )
            
            hog_row = {"sub_id": filename}
            for i, val in enumerate(hog_features):
                hog_row[f"hog_{i+1}"] = val
            dataset_hog.append(hog_row)
            
        except Exception as e:
            print(f"❌ Error en {filename}: {e}")

end_total_time = time.time()
total_execution_time = end_total_time - start_total_time

df_hog = pd.DataFrame(dataset_hog)
output_file_name = "dataset_hog.csv"
df_hog.to_csv(os.path.join(output_folder, output_file_name), index=False)

# --- IMPRESIÓN DE MÉTRICAS COMPARATIVAS ---
print("\n" + "="*50)
print("📊 COMPARACIÓN TÉCNICA - DESCRIPTOR: HOG")
print("="*50)
print(f"🔹 Número de características: {df_hog.shape[1] - 1} columnas (Vectores dependientes de la geometría celular)")
print(f"🔹 Formato de salida: {os.path.splitext(output_file_name)[1].upper()} ({output_file_name})")
print(f"🔹 Tiempo de extracción total: {total_execution_time:.4f} segundos")
print(f"🔹 Costo computacional promedio: {(total_execution_time / max(1, len(df_hog))) * 1000:.2f} ms por imagen")
print(f"🔹 Representación de datos: Vectores normalizados de gradientes orientados (Punto flotante / Float64 entre 0 y 1)")
print("="*50 + "\n")