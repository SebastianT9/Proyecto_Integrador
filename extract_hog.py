import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import hog

input_folder = "Etnias" 
output_folder = "Datasets_Descriptores"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

dataset_hog = []

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

print("🚀 Extrayendo vectores HOG (Histogram of Oriented Gradients)...")

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
            
            # Extracción HOG
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

df_hog = pd.DataFrame(dataset_hog)
df_hog.to_csv(os.path.join(output_folder, "dataset_hog.csv"), index=False)
print(f"✅ Dataset HOG guardado con éxito. Dimensiones: {df_hog.shape}")