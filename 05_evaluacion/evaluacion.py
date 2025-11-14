"""
Evaluación de Modelos
Fase 5 de CRISP-DM: Evaluación
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def cargar_modelo():
    """Carga el modelo entrenado"""
    print("Cargando modelo...")
    modelo = joblib.load('../04_modelado/modelo_riesgo_repitencia.pkl')
    print("OK Modelo cargado")
    return modelo

def cargar_datos():
    """Carga los datos de prueba"""
    print("Cargando datos de prueba...")
    X_test = pd.read_csv('../03_preparacion_datos/X_test.csv')
    y_test = pd.read_csv('../03_preparacion_datos/y_test.csv').squeeze()
    print("OK Datos cargados")
    return X_test, y_test

def evaluar_metricas(modelo, X_test, y_test):
    """Evalúa métricas del modelo"""
    print("\n" + "="*50)
    print("EVALUACIÓN DE MÉTRICAS")
    print("="*50)
    
    y_pred = modelo.predict(X_test)
    
    # Métricas generales
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nMétricas Generales:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Reporte detallado
    print("\n" + "="*50)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    print("\n" + "="*50)
    print("MATRIZ DE CONFUSIÓN")
    print("="*50)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }

def visualizar_resultados(y_test, y_pred, cm):
    """Crea visualizaciones de los resultados"""
    print("\n" + "="*50)
    print("CREANDO VISUALIZACIONES")
    print("="*50)
    
    os.makedirs('../05_evaluacion/graficos', exist_ok=True)
    
    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Alto', 'Medio', 'Bajo'],
                yticklabels=['Alto', 'Medio', 'Bajo'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig('../05_evaluacion/graficos/matriz_confusion.png', dpi=300)
    plt.close()
    print("OK Matriz de confusion guardada")
    
    # Distribución de predicciones vs reales
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    y_test_counts = pd.Series(y_test).value_counts().sort_index()
    y_pred_counts = pd.Series(y_pred).value_counts().sort_index()
    
    axes[0].bar(y_test_counts.index, y_test_counts.values, alpha=0.7, label='Real')
    axes[0].set_title('Distribución Real')
    axes[0].set_xlabel('Riesgo')
    axes[0].set_ylabel('Cantidad')
    axes[0].legend()
    
    axes[1].bar(y_pred_counts.index, y_pred_counts.values, alpha=0.7, label='Predicho', color='orange')
    axes[1].set_title('Distribución Predicha')
    axes[1].set_xlabel('Riesgo')
    axes[1].set_ylabel('Cantidad')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('../05_evaluacion/graficos/distribucion_predicciones.png', dpi=300)
    plt.close()
    print("OK Distribucion de predicciones guardada")

def analizar_errores(modelo, X_test, y_test, y_pred):
    """Analiza los errores del modelo"""
    print("\n" + "="*50)
    print("ANÁLISIS DE ERRORES")
    print("="*50)
    
    errores = X_test[y_test != y_pred].copy()
    errores['real'] = y_test[y_test != y_pred]
    errores['predicho'] = y_pred[y_test != y_pred]
    
    print(f"Total de errores: {len(errores)}")
    print(f"Tasa de error: {len(errores)/len(y_test)*100:.2f}%")
    
    print("\nDistribución de errores por clase real:")
    print(errores['real'].value_counts())
    
    print("\nDistribución de errores por clase predicha:")
    print(errores['predicho'].value_counts())
    
    # Análisis de promedios en errores
    if 'promedio_calculado' in errores.columns:
        print("\nPromedio en casos con error:")
        print(errores['promedio_calculado'].describe())
    
    return errores

def generar_reporte_final(metricas):
    """Genera un reporte final de evaluación"""
    reporte = []
    reporte.append("="*70)
    reporte.append("REPORTE FINAL DE EVALUACIÓN")
    reporte.append("="*70)
    reporte.append(f"\nFecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append("\nMÉTRICAS DE RENDIMIENTO:")
    reporte.append(f"  Accuracy:  {metricas['accuracy']:.4f}")
    reporte.append(f"  Precision: {metricas['precision']:.4f}")
    reporte.append(f"  Recall:    {metricas['recall']:.4f}")
    reporte.append(f"  F1-Score:  {metricas['f1']:.4f}")
    
    reporte.append("\n" + "="*70)
    reporte.append("CONCLUSIONES")
    reporte.append("="*70)
    
    if metricas['accuracy'] >= 0.80:
        reporte.append("OK El modelo cumple con el objetivo de precision (>80%)")
    else:
        reporte.append("⚠ El modelo no alcanza el objetivo de precisión (>80%)")
    
    reporte.append("\nEl modelo está listo para el despliegue si:")
    reporte.append("  1. Accuracy >= 0.80")
    reporte.append("  2. Las métricas por clase son balanceadas")
    reporte.append("  3. La matriz de confusión muestra buen rendimiento")
    
    reporte_texto = "\n".join(reporte)
    
    # Guardar reporte
    with open('../05_evaluacion/reporte_evaluacion.txt', 'w', encoding='utf-8') as f:
        f.write(reporte_texto)
    
    print("\n" + reporte_texto)
    print("\nOK Reporte guardado en reporte_evaluacion.txt")

def main():
    """Función principal"""
    print("="*70)
    print("EVALUACIÓN DE MODELOS")
    print("Fase 5 de CRISP-DM: Evaluación")
    print("="*70)
    
    # Cargar modelo y datos
    modelo = cargar_modelo()
    X_test, y_test = cargar_datos()
    
    # Evaluar métricas
    metricas = evaluar_metricas(modelo, X_test, y_test)
    
    # Visualizar resultados
    visualizar_resultados(y_test, metricas['y_pred'], metricas['confusion_matrix'])
    
    # Analizar errores
    errores = analizar_errores(modelo, X_test, y_test, metricas['y_pred'])
    
    # Generar reporte final
    generar_reporte_final(metricas)
    
    print("\n" + "="*70)
    print("EVALUACIÓN COMPLETADA")
    print("="*70)

if __name__ == "__main__":
    main()

