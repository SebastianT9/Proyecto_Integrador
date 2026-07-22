import cv2
import os
import numpy as np

input_folder = "Etnias" 
output_folder = "Etnias_Procesadas" 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

procesadas = 0
omitidas = 0

# Función interna para aplicar Corrección Gamma
def adjust_gamma(image, gamma=1.5):
    # Construir una tabla de consulta (LUT) para mapear los píxeles eficientemente
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

for filename in os.listdir(input_folder):
    filename_lower = filename.lower()
    
    if filename_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"⚠️ Omitida: {filename}")
                omitidas += 1
                continue
            
            # --- NUEVA ESTRATEGIA DE PREPROCESAMIENTO ---
            
            # 1. Reducción de ruido: Filtro Bilateral (Se queda igual)
            denoised_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
            
            # 2. Contraste: Cambiado Ecualización Global por Corrección Gamma (Alternativa real a CLAHE)
            # Un gamma de 1.5 - 1.8 ayuda a atenuar el brillo excesivo del fondo y resalta rasgos
            enhanced_image = adjust_gamma(denoised_image, gamma=1.5)
            
            # 3. Umbralización: Adaptive Threshold Gaussiano (Con vecindario intermedio)
            gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            
            # Ajustamos el tamaño del bloque a 31 para capturar estructuras reales del rostro
            thresholded_image = cv2.adaptiveThreshold(
                gray_image, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                31, 
                3    
            )
            
            # --- GUARDAR ---
            output_path = os.path.join(output_folder, filename)
            
            cv2.imwrite(output_path, thresholded_image)
            procesadas += 1
            
        except Exception as e:
            print(f"❌ Error en {filename}: {e}")
            omitidas += 1

print("\n--- RESUMEN DEL PROCESO ---")
print(f"✅ Procesadas con éxito: {procesadas}")