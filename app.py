import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score
import psycopg2
import time
from datetime import datetime
import os

# Conexión a PostgreSQL con manejo de excepciones
try:
    conn = psycopg2.connect(
        dbname="sensor_db",
        user="your_username",
        password="your_password",
        host="localhost",
        port="5432"
    )
except psycopg2.Error as e:
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error al conectar a la base de datos: {str(e)}")
    exit(1)

# Definir el modelo Keras
def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

while True:
    try:
        # Obtener los datos más recientes (últimos 100 registros)
        query = "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 100"
        df = pd.read_sql_query(query, conn)

        if df.empty or len(df) < 20:  # Mínimo 20 registros para entrenamiento
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] No hay suficientes datos. Esperando...")
            time.sleep(60)
            continue

        # Limpiar y convertir la columna 'fallo' a entero
        df['fallo'] = df['fallo'].apply(lambda x: int(x) if str(x).replace(' ', '').isdigit() else 0)

        # Preparar datos
        imputer = IterativeImputer(max_iter=10, random_state=42)
        scaler = StandardScaler()
        X = df[['vibracion', 'temperatura', 'corriente']]
        y = df['fallo']
        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

        # Crear y entrenar el modelo
        model = create_model(input_shape=(X_train.shape[1],))
        start_time = time.time()
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        training_time = time.time() - start_time

        # Guardar el modelo
        model.save('model.h5')
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Modelo guardado en model.h5")

        # Predicción y evaluación
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        roc_auc = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test)) > 1 else 0.5

        # Mostrar métricas en tiempo real
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Métricas:")
        print(f"- Tiempo de entrenamiento: {training_time:.2f} segundos")
        print(f"- ROC-AUC en test: {roc_auc:.3f}")

        # Mostrar predicciones en tiempo real (muestra de 5 registros)
        X_full = scaler.transform(imputer.transform(df[['vibracion', 'temperatura', 'corriente']]))
        predictions = (model.predict(X_full, verbose=0) > 0.5).astype(int)
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Predicciones (muestra de 5 registros):")
        for i in range(min(5, len(df))):
            print(f"- Timestamp: {df['timestamp'].iloc[i]}, Vibración: {df['vibracion'].iloc[i]:.2f}, "
                  f"Temperatura: {df['temperatura'].iloc[i]:.2f}, Corriente: {df['corriente'].iloc[i]:.2f}, "
                  f"Fallo: {df['fallo'].iloc[i]}, Predicción: {predictions[i][0]}")

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error: {str(e)}")
        time.sleep(60)
        continue

    # Esperar 1 minuto
    time.sleep(60)

# Cerrar conexión al finalizar
conn.close()