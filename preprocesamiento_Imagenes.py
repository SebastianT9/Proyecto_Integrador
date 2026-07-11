import cv2
import os

input_folder = "Etnias" 
output_folder = "Etnias_Procesadas" 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Contadores para saber exactamente qué pasó
procesadas = 0
omitidas = 0

for filename in os.listdir(input_folder):
    # Convertimos el nombre a minúsculas para validar la extensión sin importar si es .JPG o .jpg
    filename_lower = filename.lower()
    
    if filename_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        
        try:
            # Leer la imagen
            image = cv2.imread(input_path)
            
            # Validar que la imagen no esté vacía o corrupta
            if image is None:
                print(f"⚠️ Omitida (No se pudo leer/Corrupta): {filename}")
                omitidas += 1
                continue
            
            # --- TUS TÉCNICAS DE PREPROCESAMIENTO ---
            # 1. Reducción de ruido
            denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
            
            # 2. Contraste (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_image)
            l_clahe = clahe.apply(l)
            enhanced_image = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
            
            # 3. Umbralización
            gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            retval, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            
            # --- GUARDAR ---
            output_filename = os.path.splitext(filename)[0] + "_processed.png"
            output_path = os.path.join(output_folder, output_filename)
            
            cv2.imwrite(output_path, thresholded_image)
            procesadas += 1
            
        except Exception as e:
            print(f"❌ Error inesperado en {filename}: {e}")
            omitidas += 1
    else:
        # Archivos que no son imágenes (ej. .DS_Store, .txt)
        if not filename.startswith('.'): 
            print(f"ℹ️ Archivo ignorado (no es formato válido): {filename}")

print("\n--- RESUMEN DEL PROCESO ---")
print(f"✅ Imágenes procesadas con éxito: {procesadas}")
print(f"❌ Archivos omitidos o con error: {omitidas}")
print(f"📊 Total analizados: {procesadas + omitidas}")
