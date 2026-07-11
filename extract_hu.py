import cv2
import os
import numpy as np
import pandas as pd

input_folder = "Etnias" 
output_folder = "Datasets_Descriptores"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

dataset_hu = []

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

print("🚀 Extrayendo Momentos de Hu...")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        try:
            image = cv2.imread(input_path)
            if image is None: continue
            
            # Preprocesamiento Base (Meta 2)
            image_resized = cv2.resize(image, (128, 128))
            denoised = cv2.bilateralFilter(image_resized, d=9, sigmaColor=75, sigmaSpace=75)
            enhanced = adjust_gamma(denoised, gamma=1.5)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3
            )
            
            # Extracción de Momentos de Hu
            moments = cv2.moments(thresh)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Transformación logarítmica para normalizar la escala de los valores
            hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            hu_row = {"sub_id": filename}
            for i, val in enumerate(hu_log):
                hu_row[f"hu_{i+1}"] = val
            dataset_hu.append(hu_row)
            
        except Exception as e:
            print(f"❌ Error en {filename}: {e}")

df_hu = pd.DataFrame(dataset_hu)
df_hu.to_csv(os.path.join(output_folder, "dataset_momentos_hu.csv"), index=False)
print(f"✅ Dataset de Hu guardado con éxito. Dimensiones: {df_hu.shape}")