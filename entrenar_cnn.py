import cv2
import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
input_folder = "Etnias"
csv_folder = "Meta"  
output_folder = "Datasets_Descriptores" # Para mantener el orden de tus carpetas

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Diccionario de mapeo de clases a números
class_mapping = {
    "Mestizos.csv": 0,
    "Afro-Ecuadorians.csv": 1,
    "European_Descendants.csv": 2,
    "Indigenous.csv": 3
}

# --- 2. FUNCIÓN DE CORRECCIÓN GAMMA (TU ESTRATEGIA REPETIBLE) ---
def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- 3. CARGAR MAPEO DE ETIQUETAS DESDE LOS CSV (SUPER INGENIERÍA DE COLUMNAS) ---
print("📊 Cargando y mapeando etiquetas desde archivos CSV...")
image_labels = {}

for csv_name, class_id in class_mapping.items():
    csv_path = os.path.join(csv_folder, csv_name)
    if os.path.exists(csv_path):
        try:
            # Intento 1: Leer normal (separado por comas)
            df = pd.read_csv(csv_path)
            
            # Si se leyó mal (solo hay una columna), reintentamos con punto y coma
            if df.shape[1] <= 1:
                df = pd.read_csv(csv_path, sep=';')
                
            # Limpiamos los nombres de las columnas eliminando espacios en blanco o caracteres ocultos de Excel
            df.columns = df.columns.str.strip().str.lower()
            
            # Buscamos una columna que CONTENGA la palabra 'filename' o 'sub_id'
            col_encontrada = None
            for col in df.columns:
                if 'filename' in col or 'sub_id' in col or 'file' in col:
                    col_encontrada = col
                    break
            
            if col_encontrada is not None:
                # .unique() evita procesar filas repetidas del mismo archivo
                for filename in df[col_encontrada].dropna().unique():
                    # Guardamos el nombre limpio y en minúsculas
                    image_labels[str(filename).strip().lower()] = class_id
            else:
                # Si falla, te va a listar en pantalla las columnas que detecta para saber qué hay dentro
                print(f"⚠️ Columnas reales detectadas en {csv_name}: {list(df.columns)}")
                
        except Exception as e:
            print(f"❌ Falló la lectura del archivo {csv_name}: {e}")
    else:
        print(f"❌ No se encontró el archivo CSV: {csv_name}")
        
# --- 4. CARGAR Y PREPROCESAR LAS IMÁGENES PARA LA CNN ---
print("🚀 Cargando y preprocesando imágenes para la CNN...")
X_data = []
y_data = []

for filename in os.listdir(input_folder):
    filename_lower = filename.lower()
    if filename_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Verificar si la imagen está registrada en alguno de los CSV
        if filename_lower in image_labels:
            input_path = os.path.join(input_folder, filename)
            try:
                image = cv2.imread(input_path)
                if image is None: continue
                
                # Redimensionamiento estándar
                image_resized = cv2.resize(image, (128, 128))
                
                # Reducción de ruido y realce de contraste (Idéntico a tus descriptores)
                denoised = cv2.bilateralFilter(image_resized, d=9, sigmaColor=75, sigmaSpace=75)
                enhanced = adjust_gamma(denoised, gamma=1.5)
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                
                # Normalizar los píxeles para la red neuronal (de 0-255 a 0-1)
                gray_normalized = gray / 255.0
                
                X_data.append(gray_normalized)
                y_data.append(image_labels[filename_lower])
                
            except Exception as e:
                print(f"❌ Error al procesar {filename}: {e}")

# Convertir listas a arreglos de NumPy
X_data = np.array(X_data)
y_data = np.array(y_data)

# Añadir una dimensión de canal para indicar que es escala de grises (128, 128, 1)
X_data = np.expand_dims(X_data, axis=-1)

print(f"✅ Total de imágenes cargadas con éxito: {len(X_data)}")

# --- 5. DIVISIÓN DE DATOS (ENTREMANIENTO Y VALIDACIÓN) ---
# 80% para entrenar la red, 20% para validar resultados (sin mezclar data)
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

# --- 6. DISEÑO DE LA ARQUITECTURA DE LA CNN ---
print("🧠 Construyendo la arquitectura de la Red Neuronal Convolucional...")
model = models.Sequential([
    # Primera capa convolucional + max pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Segunda capa convolucional + max pooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Tercera capa convolucional + max pooling
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Aplanado de características
    layers.Flatten(),
    
    # Capas densas de clasificación con Dropout para evitar Overfitting
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax') # 4 neuronas de salida para las 4 etnias
])

# Compilar el modelo
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# --- 7. ENTRENAMIENTO DEL MODELO ---
print("🏋️ Entrenando la CNN...")
start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=20,          # Puedes ajustar el número de épocas según veas el rendimiento
    batch_size=32,
    validation_data=(X_val, y_val)
)

end_time = time.time()
cnn_execution_time = end_time - start_time

# --- 8. EVALUACIÓN FINAL ---
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

# --- IMPRESIÓN DE MÉTRICAS (REEMPLAZA LAS ÚLTIMAS LÍNEAS DEL SCRIPT) ---
print("\n" + "="*50)
print("📊 REPORTE DE RENDIMIENTO DIRECTO - MODELO: CNN")
print("="*50)
print(f"🔹 Exactitud Final en Entrenamiento: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"🔹 Exactitud Final en Validación (Accuracy): {val_acc * 100:.2f}%")
print(f"🔹 Pérdida Final (Loss): {val_loss:.4f}")
print("="*50 + "\n")

# Guardar el modelo entrenado para futuras comparaciones
model.save(os.path.join(output_folder, "modelo_etnias_cnn.h5"))
print(f"💾 Modelo guardado exitosamente en '{output_folder}/modelo_etnias_cnn.h5'")
