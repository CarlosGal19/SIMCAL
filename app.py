import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score  # Añadido cross_val_score
from sklearn.experimental import enable_iterative_imputer  # Habilita IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sqlite3
import time
from datetime import datetime

# Conexión a las bases de datos con manejo de excepciones
try:
    db_path = "simulated_data.db"
    conn = sqlite3.connect(db_path)
    output_db_path = "predictions.db"
    output_conn = sqlite3.connect(output_db_path)
    cursor = output_conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions
                      (timestamp TEXT, vibracion REAL, temperatura REAL, corriente REAL, fallo INTEGER, prediccion INTEGER)''')
except sqlite3.Error as e:
    print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error al conectar a la base de datos: {str(e)}")
    exit(1)

while True:
    try:
        # Obtener los datos más recientes (últimos 100 registros)
        query = "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 100"
        df = pd.read_sql_query(query, conn)

        if df.empty or len(df) < 20:  # Mínimo 20 registros para entrenamiento
            print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] No hay suficientes datos. Esperando...")
            time.sleep(60)
            continue

        # Limpiar y convertir la columna 'fallo' a entero, manejando valores corruptos
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

        # Definir modelos ensamblados
        rf_model = RandomForestClassifier(random_state=42)
        xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')  # Eliminado use_label_encoder
        ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft')

        # Optimización de hiperparámetros
        param_dist = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [10, 20],
            'xgb__n_estimators': [100, 200],
            'xgb__max_depth': [3, 5]
        }
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(ensemble_model, param_dist, n_iter=5, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1, error_score='raise')
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        print(f"\n[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Mejores hiperparámetros: {random_search.best_params_}")

        # Entrenamiento y tiempo
        start_time = time.time()
        best_model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predicción y evaluación
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test)) > 1 else 0.5  # Evitar división por cero
        cv_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='roc_auc')
        cv_mean = cv_scores.mean()

        # Mostrar métricas en tiempo real
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Métricas:")
        print(f"- Tiempo de entrenamiento: {training_time:.2f} segundos")
        print(f"- ROC-AUC en test: {roc_auc:.3f}")
        print(f"- ROC-AUC CV (media): {cv_mean:.3f}")

        # Mostrar predicciones en tiempo real (muestra de 5 registros)
        X_full = scaler.transform(imputer.transform(df[['vibracion', 'temperatura', 'corriente']]))
        predictions = best_model.predict(X_full)
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Predicciones (muestra de 5 registros):")
        for i in range(min(5, len(df))):
            print(f"- Timestamp: {df['timestamp'].iloc[i]}, Vibración: {df['vibracion'].iloc[i]:.2f}, "
                  f"Temperatura: {df['temperatura'].iloc[i]:.2f}, Corriente: {df['corriente'].iloc[i]:.2f}, "
                  f"Fallo: {df['fallo'].iloc[i]}, Predicción: {predictions[i]}")

        # Guardar predicciones en la base de datos
        df['prediccion'] = predictions
        df.to_sql('predictions', output_conn, if_exists='append', index=False)
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Predicciones guardadas en {output_db_path}")

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y%m%d_%H%M%S')}] Error: {str(e)}")
        time.sleep(60)
        continue

    # Esperar 1 minuto
    time.sleep(60)

# Cerrar conexiones al finalizar
conn.close()
output_conn.close()