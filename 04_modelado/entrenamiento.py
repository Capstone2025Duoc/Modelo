"""
Entrenamiento de Modelos
Fase 4 de CRISP-DM: Modelado
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import os

SAMPLE_FRAC = 0.35  # reduce dataset size for faster experimentation
MAX_TRAIN_SAMPLES = 200_000
CV_FOLDS = 3


def cargar_datos():
    """Carga los datos preparados"""
    print("Cargando datos preparados...")
    X_train = pd.read_csv('../03_preparacion_datos/X_train.csv')
    X_test = pd.read_csv('../03_preparacion_datos/X_test.csv')
    y_train = pd.read_csv('../03_preparacion_datos/y_train.csv').squeeze()
    y_test = pd.read_csv('../03_preparacion_datos/y_test.csv').squeeze()
    
    if SAMPLE_FRAC < 1.0 or len(X_train) > MAX_TRAIN_SAMPLES:
        frac = SAMPLE_FRAC if len(X_train) * SAMPLE_FRAC <= MAX_TRAIN_SAMPLES else MAX_TRAIN_SAMPLES / len(X_train)
        sample_idx = X_train.sample(frac=frac, random_state=42).index
        X_train = X_train.loc[sample_idx]
        y_train = y_train.loc[sample_idx]
        print(f"Entrenamiento (muestreado): {X_train.shape}")
    else:
        print(f"Entrenamiento: {X_train.shape}")
    print(f"Prueba: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def entrenar_modelos(X_train, y_train):
    """Entrena múltiples modelos y selecciona el mejor"""
    print("\n" + "="*50)
    print("ENTRENAMIENTO DE MODELOS")
    print("="*50)
    
    modelos = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=500, multi_class='multinomial')
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f"\nEntrenando {nombre}...")
        
        # Validación cruzada
        scores = cross_val_score(modelo, X_train, y_train, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
        
        # Entrenar en todo el conjunto
        modelo.fit(X_train, y_train)
        
        resultados[nombre] = {
            'modelo': modelo,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'scores': scores
        }
        
        print(f"  CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    return resultados

def optimizar_hiperparametros(X_train, y_train):
    """Optimiza hiperparámetros del mejor modelo"""
    print("\n" + "="*50)
    print("OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*50)
    
    # Usar RandomForest como base (generalmente funciona bien)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    print("Buscando mejores hiperparámetros...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=CV_FOLDS, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score CV: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluar_modelo(modelo, X_test, y_test):
    """Evalúa el modelo en el conjunto de prueba"""
    print("\n" + "="*50)
    print("EVALUACIÓN EN CONJUNTO DE PRUEBA")
    print("="*50)
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy, y_pred

def guardar_modelo(modelo, nombre='modelo_riesgo_repitencia'):
    """Guarda el modelo entrenado"""
    os.makedirs('../04_modelado', exist_ok=True)
    ruta = f'../04_modelado/{nombre}.pkl'
    joblib.dump(modelo, ruta)
    print(f"\nOK Modelo guardado en {ruta}")
    return ruta

def main():
    """Función principal"""
    print("="*70)
    print("ENTRENAMIENTO DE MODELOS")
    print("Fase 4 de CRISP-DM: Modelado")
    print("="*70)
    
    # Cargar datos
    X_train, X_test, y_train, y_test = cargar_datos()
    
    # Entrenar modelos
    resultados = entrenar_modelos(X_train, y_train)
    
    # Seleccionar mejor modelo basado en CV
    mejor_nombre = max(resultados.keys(), key=lambda k: resultados[k]['cv_mean'])
    print(f"\n{'='*50}")
    print(f"MEJOR MODELO (CV): {mejor_nombre}")
    print(f"Score CV: {resultados[mejor_nombre]['cv_mean']:.4f}")
    print(f"{'='*50}")
    
    # Optimizar hiperparámetros
    modelo_optimizado = optimizar_hiperparametros(X_train, y_train)
    
    # Evaluar modelo optimizado
    accuracy, y_pred = evaluar_modelo(modelo_optimizado, X_test, y_test)
    
    # Guardar modelo
    ruta_modelo = guardar_modelo(modelo_optimizado)
    
    # Guardar también el mejor modelo sin optimizar para comparación
    guardar_modelo(resultados[mejor_nombre]['modelo'], 'modelo_sin_optimizar')
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"Modelo final guardado en: {ruta_modelo}")
    print(f"Accuracy en prueba: {accuracy:.4f}")

if __name__ == "__main__":
    main()

