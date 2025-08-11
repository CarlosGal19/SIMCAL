import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score
import time
from datetime import datetime
from flask import Flask, request, jsonify
import json

# =============================================================================
# # Configuración de la conexión a PostgreSQL (comentada temporalmente)
# '''
# try:
#     conn = psycopg2.connect(
#         dbname="sensor_db",
#         user="postgres",
#         password="root",
#         host="localhost",
#         port="5432"
#     )
#     cursor = conn.cursor()
#     cursor.execute('''CREATE TABLE IF NOT EXISTS sensor_data (
#                         accel_x REAL, accel_y REAL, accel_z REAL, rms REAL,
#                         temperature REAL, humidity REAL, fallo INTEGER)''')
#     conn.commit()
# except psycopg2.Error as e:
#     print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error al conectar a la base de datos: {str(e)}")
#     exit(1)
# '''
# 
# =============================================================================

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

# Función para procesar y cargar datos desde JSON
def load_sensor_data():
    # Cargar datos desde los archivos JSON
    with open('ESP32-F5C77C.json', 'r') as f:
        aggregate_data = json.load(f)
    with open('raw_ESP32-F5C77C.json', 'r') as f:
        raw_data = json.load(f)

    # Procesar datos agregados
    agg_records = []
    for entry in aggregate_data:
        if entry['type'] == 'aggregate':
            data = entry['data']
            agg_records.append((
                data.get('MPU6050_accel_x_mean', 0),
                data.get('MPU6050_accel_y_mean', 0),
                data.get('MPU6050_accel_z_mean', 0),
                data.get('MPU6050_rms_mean', 0),
                data.get('BME280_temperature_mean', 0),
                data.get('BME280_humidity_mean', 0),
                1 if any(a['type'] == 'alert' for a in aggregate_data) else 0  # Falla si hay alerta
            ))

    # Procesar datos crudos
    raw_records = []
    for entry in raw_data:
        if entry['type'] == 'raw':
            mpu6050 = entry['sensors']['MPU6050']
            bme280 = entry['sensors']['BME280']
            raw_records.append((
                mpu6050['accel_x'],
                mpu6050['accel_y'],
                mpu6050['accel_z'],
                mpu6050['rms'],
                bme280['temperature'],
                bme280['humidity'],
                1 if mpu6050['rms'] > 10.0 else 0  # Falla si RMS > 10.0
            ))

    # Combinar y eliminar duplicados (simplificación)
    all_records = agg_records + raw_records
    unique_records = list(dict.fromkeys(tuple(x) for x in all_records))  # Eliminar duplicados

    # Crear DataFrame
    df = pd.DataFrame(unique_records, columns=['accel_x', 'accel_y', 'accel_z', 'rms', 'temperature', 'humidity', 'fallo'])
    return df

# Función para entrenar y guardar el modelo
def train_and_save_model():
    global imputer, scaler
    try:
        # Cargar datos desde JSON
        df = load_sensor_data()

        if df.empty or len(df) < 20:  # Mínimo 20 registros para entrenamiento
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] No hay suficientes datos para entrenamiento.")
            return

        # Preparar datos
        imputer = IterativeImputer(max_iter=10, random_state=42)
        scaler = StandardScaler()
        X = df[['accel_x', 'accel_y', 'accel_z', 'rms', 'temperature', 'humidity']]
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
        X_full = scaler.transform(imputer.transform(df[['accel_x', 'accel_y', 'accel_z', 'rms', 'temperature', 'humidity']]))
        predictions = (model.predict(X_full, verbose=0) > 0.5).astype(int)
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Predicciones (muestra de 5 registros):")
        for i in range(min(5, len(df))):
            print(f"- accel_x: {df['accel_x'].iloc[i]:.2f}, accel_y: {df['accel_y'].iloc[i]:.2f}, accel_z: {df['accel_z'].iloc[i]:.2f}, "
                  f"rms: {df['rms'].iloc[i]:.2f}, temperature: {df['temperature'].iloc[i]:.2f}, humidity: {df['humidity'].iloc[i]:.2f}, "
                  f"Fallo: {df['fallo'].iloc[i]}, Predicción: {predictions[i][0]}")

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error en entrenamiento: {str(e)}")

# Inicializar Flask
app = Flask(__name__)

# Inicializar transformadores globales (se actualizarán con los del entrenamiento)
imputer = IterativeImputer(max_iter=10, random_state=42)
scaler = StandardScaler()

# Entrenar el modelo al iniciar el script (ejecución manual)
if __name__ == '__main__':
    train_and_save_model()
    app.run(debug=True, host='0.0.0.0', port=5000)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del request
        data = request.get_json()
        if not data or 'accel_x' not in data or 'accel_y' not in data or 'accel_z' not in data or 'rms' not in data or 'temperature' not in data or 'humidity' not in data:
            return jsonify({'error': 'Datos incompletos. Se requieren accel_x, accel_y, accel_z, rms, temperature y humidity.'}), 400

        # Convertir datos a formato numpy
        input_data = np.array([[data['accel_x'], data['accel_y'], data['accel_z'], data['rms'], data['temperature'], data['humidity']]])

        # Cargar el modelo más reciente
        model = keras.models.load_model('model.h5')

        # Imputar y escalar datos usando los transformadores globales
        input_data_imputed = imputer.transform(input_data)
        input_data_scaled = scaler.transform(input_data_imputed)

        # Realizar predicción
        prediction_prob = model.predict(input_data_scaled, verbose=0)
        prediction = (prediction_prob > 0.5).astype(int)[0][0]

        return jsonify({'prediction': int(prediction), 'probability': float(prediction_prob[0][0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500