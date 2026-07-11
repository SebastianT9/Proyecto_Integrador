import cv2
import os
import numpy as np
import pandas as pd

input_folder = "Etnias" 
output_folder = "Datasets_Descriptores"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

dataset_orb = []

# Forzamos a extraer exactamente 50 puntos para homogeneizar las dimensiones del CSV
orb_detector = cv2.ORB_create(nfeatures=50)

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

print("🚀 Extrayendo puntos clave y descriptores ORB...")

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
            
            # Extracción ORB
            keypoints, descriptors = orb_detector.detectAndCompute(gray, None)
            
            fixed_descriptor_length = 50 * 32  # 50 keypoints * 32 bytes = 1600 columnas
            if descriptors is not None:
                orb_features = descriptors.flatten()
                # Ajuste dinámico de longitud (Zero-padding si faltan puntos)
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

df_orb = pd.DataFrame(dataset_orb)
df_orb.to_csv(os.path.join(output_folder, "dataset_orb.csv"), index=False)
print(f"✅ Dataset ORB guardado con éxito. Dimensiones: {df_orb.shape}")