import os
import numpy as np
import pandas as pd
import time
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, silhouette_score, 
                             adjusted_rand_score, adjusted_mutual_info_score)

# --- 1. CONFIGURACIÓN DE RUTAS ---
descriptors_folder = "Datasets_Descriptores"
datasets_names = ["HOG", "Hu", "ORB"]

# Estructuras para almacenar los reportes finales
resultados_clasificacion = []
resultados_clustering = []

print("🚀 INICIANDO PROCESAMIENTO: META 2 & META 3 (Árboles de Decisión + Perceptrón)...")

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
        print(f"⚠️ Saltando {name}: No se encontraron las matrices .npy.")
        continue
        
    # Carga de datos
    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    # Escalado de características (Crucial para el Perceptrón y Clustering)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Guardar el escalador en disco
    joblib.dump(scaler, os.path.join(descriptors_folder, f"scaler_{name}.pkl"))
    
    # Unir datos escalados para el clustering no supervisado
    X_all_scaled = np.vstack((X_train_scaled, X_test_scaled))
    y_all = np.concatenate((y_train, y_test))

    # =====================================================================
    # PARTE A: CLASIFICACIÓN SUPERVISADA (Árbol de Decisión & Perceptrón)
    # =====================================================================
    modelos_clasificacion = {
        # 1. Árbol de Decisión (con restricción de profundidad para evitar overfitting)
        "Arbol de Decision": DecisionTreeClassifier(
            criterion='entropy',
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        ),
        
        # 2. Perceptrón Multicapa (MLP / Perceptrón)
        "Perceptron Multicapa": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            learning_rate='adaptive',
            early_stopping=True,
            random_state=42
        )
    }
    
    for model_name, clf in modelos_clasificacion.items():
        start_time = time.time()
        
        # Entrenamiento
        clf.fit(X_train_scaled, y_train)
        execution_time = time.time() - start_time
        
        # Predicción
        y_pred = clf.predict(X_test_scaled)
        
        print(f"\n📊 REPORT DETALLADO POR CLASE PARA [{model_name}] CON [{name}]:")
        clases_nombres = ["Mestizos", "Afro-Ecuadorians", "European_Descendants", "Indigenous"]
        print(classification_report(y_test, y_pred, target_names=clases_nombres, zero_division=0))
        
        # Guardar el clasificador entrenado
        safe_model_name = model_name.lower().replace(" ", "_").replace("á", "a").replace("ó", "o")
        joblib.dump(clf, os.path.join(descriptors_folder, f"modelo_{name}_{safe_model_name}.pkl"))
        
        # Cálculo de las 4 métricas obligatorias
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        resultados_clasificacion.append({
            "Dataset": name, "Algoritmo": model_name, "Tiempo (s)": round(execution_time, 4),
            "Accuracy (%)": round(acc * 100, 2), "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2), "F1-Score (%)": round(f1 * 100, 2)
        })
        print(f"✅ Clasificador [{model_name}] evaluado y guardado con éxito.")

    # =====================================================================
    # PARTE B: CLUSTERING NO SUPERVISADO (K-Means)
    # =====================================================================
    print(f"\n🌀 Ejecutando K-Means Clustering para {name}...")
    start_time = time.time()
    
    kmeans = KMeans(
        n_clusters=4, 
        init='k-means++', 
        n_init=15, 
        random_state=42
    )
    cluster_labels = kmeans.fit_predict(X_all_scaled)
    execution_time_cluster = time.time() - start_time
    
    # Guardar modelo de clustering
    joblib.dump(kmeans, os.path.join(descriptors_folder, f"modelo_{name}_kmeans.pkl"))
    
    # Métricas de evaluación interna y externa
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