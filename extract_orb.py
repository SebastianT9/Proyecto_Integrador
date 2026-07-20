import cv2
import os
import numpy as np
import pandas as pd
import time

input_folder = "Etnias" 
output_folder = "Datasets_Descriptores"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

dataset_orb = []
# ScoreType FAST mejora la consistencia en la detección de esquinas faciales estables
orb_detector = cv2.ORB_create(nfeatures=50, scoreType=cv2.ORB_FAST_SCORE)

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

print("🚀 Extrayendo descriptores locales ORB...")
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
            
            keypoints, descriptors = orb_detector.detectAndCompute(gray, None)
            fixed_descriptor_length = 50 * 32  
            
            if descriptors is not None:
                orb_features = descriptors.flatten()
                if len(orb_features) < fixed_descriptor_length:
                    orb_features = np.pad(orb_features, (0, fixed_descriptor_length - len(orb_features)), 'constant')
                elif len(orb_features) > fixed_descriptor_length:
                    orb_features = orb_features[:fixed_descriptor_length]
            else:
                orb_features = np.zeros(fixed_descriptor_length, dtype=np.uint8)
                
            orb_row = {"sub_id": filename}
            for i, val in enumerate(orb_features):
                orb_row[f"orb_{i+1}"] = val
            dataset_orb.append(orb_row)
            
        except Exception as e:
            print(f"❌ Error en {filename}: {e}")

end_total_time = time.time()
total_execution_time = end_total_time - start_total_time

df_orb = pd.DataFrame(dataset_orb)
output_file_name = "dataset_orb.csv"
df_orb.to_csv(os.path.join(output_folder, output_file_name), index=False)

print("\n" + "="*50)
print("📊 REPORTE TÉCNICO COMPLETO - DESCRIPTOR: ORB")
print("="*50)
print(f"🔹 Dimensiones del dataset: {df_orb.shape[0]} muestras x {df_orb.shape[1] - 1} características")
print(f"🔹 Tiempo total: {total_execution_time:.4f} segundos")
print("="*50 + "\n")