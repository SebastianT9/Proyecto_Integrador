import cv2
import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2

# --- 1. CONFIGURACIÓN DE RUTAS Y MAPEO ESTRICTO ---
input_folder = "Etnias"
csv_folder = "Meta"
output_folder = "Datasets_Descriptores"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

class_names = ["Mestizos", "Afro-Ecuadorians", "European_Descendants", "Indigenous"]
class_mapping = {
    "Mestizos.csv": 0,
    "Afro-Ecuadorians.csv": 1,
    "European_Descendants.csv": 2,
    "Indigenous.csv": 3
}

# --- 2. FUNCIÓN DE PREPROCESAMIENTO ---
def adjust_gamma_rgb(image, gamma=1.3):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- 3. CARGAR ETIQUETAS ---
print("📊 Cargando y mapeando etiquetas desde archivos CSV...")
image_labels = {}

for csv_name, class_id in class_mapping.items():
    csv_path = os.path.join(csv_folder, csv_name)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if df.shape[1] <= 1:
                df = pd.read_csv(csv_path, sep=';')
                
            df.columns = df.columns.str.strip().str.lower()
            col_encontrada = next((col for col in df.columns if any(k in col for k in ['filename', 'sub_id', 'file'])), None)
            
            if col_encontrada is not None:
                for filename in df[col_encontrada].dropna().unique():
                    clean_name = os.path.splitext(str(filename).strip().lower())[0]
                    image_labels[clean_name] = class_id
        except Exception as e:
            print(f"❌ Error al leer {csv_name}: {e}")

# --- 4. CARGAR Y PREPROCESAR IMÁGENES ---
print("🚀 Cargando imágenes RGB...")
X_list, y_list = [], []

for filename in os.listdir(input_folder):
    filename_lower = filename.lower()
    clean_id = os.path.splitext(filename_lower)[0]
    
    if filename_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp')) and clean_id in image_labels:
        input_path = os.path.join(input_folder, filename)
        try:
            image = cv2.imread(input_path)
            if image is None: continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (128, 128))
            
            # NOTA: Se eliminó el bilateralFilter para NO destruir las texturas faciales (arrugas, poros) 
            # que son vitales para que la CNN diferencie etnias.
            enhanced = adjust_gamma_rgb(image_resized, gamma=1.3)
            
            X_list.append(enhanced / 255.0)
            y_list.append(image_labels[clean_id])
        except Exception as e:
            print(f"❌ Error en {filename}: {e}")

X_data = np.array(X_list)
y_data = np.array(y_list)

# --- 5. SPLIT ESTRATIFICADO ---
X_train_raw, X_val, y_train_raw, y_val = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# --- 6. MANEJO DE DESBALANCE CON PESOS DE CLASE ---
print("\n⚖️ Calculando pesos de clase para manejar el desbalance...")
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train_raw), 
    y=y_train_raw
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Pesos asignados: {class_weight_dict}")

# --- 7. ARQUITECTURA DE CNN OPTIMIZADA Y ROBUSTA ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.1)
])

# 1. Cargamos el modelo base preentrenado (sin la capa final)
base_model = MobileNetV2(
    input_shape=(128, 128, 3), 
    include_top=False, 
    weights='imagenet'
)

# 2. Congelamos el modelo base para que no olvide lo que ya sabe
base_model.trainable = False 

# 3. Construimos nuestro modelo conectando las partes
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(), # Aplanado especializado para Transfer Learning
    layers.Dropout(0.4),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Learning rate un poco más alto
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

# --- 8. ENTRENAMIENTO ---
print("\n🏋️ Entrenando la CNN...")
start_time = time.time()

history = model.fit(
    X_train_raw, y_train_raw, 
    epochs=50, 
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    class_weight=class_weight_dict, 
    verbose=1
)

end_time = time.time()
print(f"✅ Entrenamiento completado en {round((end_time - start_time)/60, 2)} minutos.")

# --- 9. EVALUACIÓN Y MATRIZ REAL ---
y_pred_probs = model.predict(X_val, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n" + "="*50)
print("📊 REPORTE DE EVALUACIÓN OPTIMIZADO - CNN")
print("="*50)
print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))

print("🧱 MATRIZ DE CONFUSIÓN:")
print(confusion_matrix(y_val, y_pred))
print("="*50 + "\n")

model.save(os.path.join(output_folder, "modelo_etnias_cnn_equilibrado.keras"))
print(f"💾 Modelo guardado exitosamente en: {output_folder}")