import os
import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             silhouette_score, adjusted_rand_score, adjusted_mutual_info_score)

# --- 1. CONFIGURACIÓN DE RUTA ---
descriptors_folder = "Datasets_Descriptores"
datasets_names = ["HOG", "Hu", "ORB"]

# Estructuras para almacenar los reportes finales
resultados_clasificacion = []
resultados_clustering = []

print("INICIANDO PROCESAMIENTO MASIVO: META 2 & META 3...")

for name in datasets_names:
    print(f"\n" + "="*60)
    print(f"ENFOQUE DESCRIPTOR: {name}")
    print("="*60)
    
    # Cargar matrices generadas en la Meta 1
    x_train_path = os.path.join(descriptors_folder, f"X_train_{name}.npy")
    x_test_path = os.path.join(descriptors_folder, f"X_test_{name}.npy")
    y_train_path = os.path.join(descriptors_folder, f"y_train_{name}.npy")
    y_test_path = os.path.join(descriptors_folder, f"y_test_{name}.npy")
    
    if not (os.path.exists(x_train_path) and os.path.exists(x_test_path)):
        print(f"Saltando {name}: No se encontraron las matrices .npy.")
        continue
        
  #Carga de datos igual que antes
    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    # NUEVO: ESCALADO DE CARACTERÍSTICAS (SVM y Clustering)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Unimos el set completo ESCALADO para Clustering
    X_all_scaled = np.vstack((X_train_scaled, X_test_scaled))
    y_all = np.concatenate((y_train, y_test))

    # PARTE A: CLASIFICACIÓN (Con datos escalados)
    modelos_clasificacion = {
    # 🌟 Agregamos class_weight='balanced' a ambos modelos
    "SVM (RBF)": SVC(kernel='rbf', C=1.0, random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }
    
    for model_name, clf in modelos_clasificacion.items():
        start_time = time.time()
        
        # Entrenamos con los datos optimizados y escalados
        clf.fit(X_train_scaled, y_train)
        execution_time = time.time() - start_time
        
        y_pred = clf.predict(X_test_scaled)
        
        # Las 4 métricas de clasificación requeridas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        resultados_clasificacion.append({
            "Dataset": name, "Algoritmo": model_name, "Tiempo (s)": round(execution_time, 4),
            "Accuracy (%)": round(acc * 100, 2), "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2), "F1-Score (%)": round(f1 * 100, 2)
        })
        print(f"Clasificador [{model_name}] evaluado con éxito.")

    # PARTE B: CLUSTERING (Con datos escalados)
   
    print(f"Ejecutando K-Means Clustering...")
    start_time = time.time()
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_all_scaled)
    
    execution_time_cluster = time.time() - start_time
    
    ari = adjusted_rand_score(y_all, cluster_labels)
    ami = adjusted_mutual_info_score(y_all, cluster_labels)
    
    try:
        silueta = silhouette_score(X_all_scaled, cluster_labels)
    except:
        silueta = 0.0
        
    resultados_clustering.append({
        "Dataset": name, "Algoritmo": "K-Means (k=4)", "Tiempo (s)": round(execution_time_cluster, 4),
        "Silueta (Interna)": round(silueta, 4), "ARI (Externa)": round(ari, 4), "AMI (Externa)": round(ami, 4)
    })

# --- PRESENTACIÓN DE TABLAS FINALES ---
df_class = pd.DataFrame(resultados_clasificacion)
df_clust = pd.DataFrame(resultados_clustering)

print("\n" + "═"*85)
print("META 3: CUADRO COMPARATIVO - RENDIMIENTO DE CLASIFICACIÓN")
print("═"*85)
print(df_class.to_string(index=False))
print("═"*85)

print("\n" + "═"*85)
print("META 3: CUADRO COMPARATIVO - RENDIMIENTO DE CLUSTERING")
print("═"*85)
print(df_clust.to_string(index=False))
print("═"*85 + "\n")