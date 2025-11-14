"""
Preparación de Datos
Fase 3 de CRISP-DM: Preparación de los Datos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def cargar_datos():
    """Carga los datos del EDA"""
    print("Cargando datos del EDA...")
    try:
        df = pd.read_csv('../02_comprension_datos/datos_eda.csv')
    except:
        # Si no existe, cargar directamente del CSV original
        df = pd.read_csv(
            '../20230313_Notas_y_Egresados_Enseñanza_Media_2024_PUBL.csv',
            sep=';',
            decimal=',',
            encoding='utf-8'
        )
        # Convertir promedio a numérico
        if df['PROM_NOTAS_ALU'].dtype == 'object':
            df['PROM_NOTAS_ALU'] = df['PROM_NOTAS_ALU'].str.replace(',', '.').astype(float)
    
    print(f"Datos cargados: {len(df)} registros")
    return df

def limpiar_datos(df):
    """Limpieza de datos"""
    print("\n" + "="*50)
    print("LIMPIEZA DE DATOS")
    print("="*50)
    
    print(f"Registros iniciales: {len(df)}")
    
    # Eliminar valores nulos en PROM_NOTAS_ALU
    df = df.dropna(subset=['PROM_NOTAS_ALU'])
    print(f"Después de eliminar nulos en PROM_NOTAS_ALU: {len(df)}")
    
    # Filtrar valores fuera del rango válido (1.0 - 7.0)
    df = df[(df['PROM_NOTAS_ALU'] >= 1.0) & (df['PROM_NOTAS_ALU'] <= 7.0)]
    print(f"Después de filtrar rango válido (1.0-7.0): {len(df)}")
    
    return df

def crear_variable_objetivo(df):
    """Crea la variable objetivo de riesgo de repitencia"""
    print("\n" + "="*50)
    print("CREACIÓN DE VARIABLE OBJETIVO")
    print("="*50)
    
    # Crear variable objetivo basada en el promedio
    # Alto riesgo: < 3.5
    # Medio riesgo: 3.5 - 3.9
    # Bajo riesgo: >= 4.0
    
    def clasificar_riesgo(promedio):
        if promedio < 3.5:
            return 'alto'
        elif promedio < 4.0:
            return 'medio'
        else:
            return 'bajo'
    
    df['RIESGO'] = df['PROM_NOTAS_ALU'].apply(clasificar_riesgo)
    
    print("Distribución de riesgo:")
    print(df['RIESGO'].value_counts())
    print("\nPorcentajes:")
    print(df['RIESGO'].value_counts(normalize=True) * 100)
    
    return df

def crear_features_simuladas(df):
    """
    Crea features simuladas para entrenar el modelo
    Simula el escenario donde tenemos hasta 3 notas parciales
    """
    print("\n" + "="*50)
    print("CREACIÓN DE FEATURES SIMULADAS")
    print("="*50)
    
    # Para cada estudiante, simulamos que tenemos 1, 2 o 3 notas parciales
    # que al promediarse dan el PROM_NOTAS_ALU
    
    np.random.seed(42)
    
    datos_simulados = []
    
    for idx, row in df.iterrows():
        promedio_final = row['PROM_NOTAS_ALU']
        
        # Simular diferentes escenarios de notas parciales
        # Escenario 1: 1 nota (es el promedio mismo)
        datos_simulados.append({
            'nota_1': promedio_final,
            'nota_2': np.nan,
            'nota_3': np.nan,
            'promedio': promedio_final,
            'riesgo': row['RIESGO']
        })
        
        # Escenario 2: 2 notas que promedian al promedio final
        # Generar dos notas que promedien al promedio final
        nota1 = np.random.uniform(max(1.0, promedio_final - 1.5), min(7.0, promedio_final + 1.5))
        nota2 = 2 * promedio_final - nota1
        nota2 = np.clip(nota2, 1.0, 7.0)
        # Recalcular promedio real
        promedio_real = (nota1 + nota2) / 2
        
        datos_simulados.append({
            'nota_1': nota1,
            'nota_2': nota2,
            'nota_3': np.nan,
            'promedio': promedio_real,
            'riesgo': clasificar_riesgo(promedio_real)
        })
        
        # Escenario 3: 3 notas que promedian al promedio final
        nota1 = np.random.uniform(max(1.0, promedio_final - 1.5), min(7.0, promedio_final + 1.5))
        nota2 = np.random.uniform(max(1.0, promedio_final - 1.5), min(7.0, promedio_final + 1.5))
        nota3 = 3 * promedio_final - nota1 - nota2
        nota3 = np.clip(nota3, 1.0, 7.0)
        promedio_real = (nota1 + nota2 + nota3) / 3
        
        datos_simulados.append({
            'nota_1': nota1,
            'nota_2': nota2,
            'nota_3': nota3,
            'promedio': promedio_real,
            'riesgo': clasificar_riesgo(promedio_real)
        })
    
    df_features = pd.DataFrame(datos_simulados)
    
    print(f"Features creadas: {len(df_features)} registros")
    print(f"\nDistribución de cantidad de notas:")
    print(f"  1 nota: {(df_features['nota_2'].isna()).sum()}")
    print(f"  2 notas: {((df_features['nota_2'].notna()) & (df_features['nota_3'].isna())).sum()}")
    print(f"  3 notas: {(df_features['nota_3'].notna()).sum()}")
    
    return df_features

def crear_features_derivadas(df):
    """Crea features derivadas de las notas"""
    print("\n" + "="*50)
    print("CREACIÓN DE FEATURES DERIVADAS")
    print("="*50)
    
    # Promedio de las notas disponibles
    df['promedio_calculado'] = df[['nota_1', 'nota_2', 'nota_3']].mean(axis=1, skipna=True)
    
    # Cantidad de notas disponibles
    df['cantidad_notas'] = df[['nota_1', 'nota_2', 'nota_3']].notna().sum(axis=1)
    
    # Tendencia (mejora o empeora)
    def calcular_tendencia(row):
        notas = [row['nota_1'], row['nota_2'], row['nota_3']]
        notas = [n for n in notas if not pd.isna(n)]
        if len(notas) >= 2:
            # Comparar primera vs última
            if notas[-1] > notas[0]:
                return 1  # Mejora
            elif notas[-1] < notas[0]:
                return -1  # Empeora
            else:
                return 0  # Estable
        return 0
    
    df['tendencia'] = df.apply(calcular_tendencia, axis=1)
    
    # Variabilidad (desviación estándar)
    def calcular_variabilidad(row):
        notas = [row['nota_1'], row['nota_2'], row['nota_3']]
        notas = [n for n in notas if not pd.isna(n)]
        if len(notas) >= 2:
            return np.std(notas)
        return 0
    
    df['variabilidad'] = df.apply(calcular_variabilidad, axis=1)
    
    # Nota mínima y máxima
    df['nota_min'] = df[['nota_1', 'nota_2', 'nota_3']].min(axis=1, skipna=True)
    df['nota_max'] = df[['nota_1', 'nota_2', 'nota_3']].max(axis=1, skipna=True)
    
    # Distancia al umbral de aprobación
    df['distancia_umbral'] = df['promedio_calculado'] - 4.0
    
    print("Features creadas:")
    print(df[['promedio_calculado', 'cantidad_notas', 'tendencia', 'variabilidad', 
              'nota_min', 'nota_max', 'distancia_umbral']].describe())
    
    return df

def preparar_datos_modelo(df):
    """Prepara los datos para el modelo"""
    print("\n" + "="*50)
    print("PREPARACIÓN PARA MODELO")
    print("="*50)
    
    # Seleccionar features (SIN promedio_calculado ni distancia_umbral para evitar data leakage)
    # El modelo debe aprender de las notas individuales, no del promedio calculado
    features = ['nota_1', 'nota_2', 'nota_3', 
                'cantidad_notas', 'tendencia', 'variabilidad', 
                'nota_min', 'nota_max']
    
    X = df[features].copy()
    
    # Rellenar NaN con 0 para las notas no disponibles
    X = X.fillna(0)
    
    # Variable objetivo
    y = df['riesgo']
    
    print(f"Features: {X.shape}")
    print(f"Target: {y.shape}")
    print(f"\nDistribución de clases:")
    print(y.value_counts())
    
    return X, y

def dividir_datos(X, y):
    """Divide los datos en entrenamiento y prueba"""
    print("\n" + "="*50)
    print("DIVISIÓN DE DATOS")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape}")
    print(f"Conjunto de prueba: {X_test.shape}")
    print(f"\nDistribución en entrenamiento:")
    print(y_train.value_counts())
    print(f"\nDistribución en prueba:")
    print(y_test.value_counts())
    
    return X_train, X_test, y_train, y_test

def main():
    """Función principal"""
    print("="*70)
    print("PREPARACIÓN DE DATOS")
    print("Fase 3 de CRISP-DM: Preparación de los Datos")
    print("="*70)
    
    # Cargar datos
    df = cargar_datos()
    
    # Limpiar datos
    df = limpiar_datos(df)
    
    # Crear variable objetivo
    df = crear_variable_objetivo(df)
    
    # Crear features simuladas
    df_features = crear_features_simuladas(df)
    
    # Crear features derivadas
    df_features = crear_features_derivadas(df_features)
    
    # Preparar datos para modelo
    X, y = preparar_datos_modelo(df_features)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    
    # Guardar datos procesados
    os.makedirs('../03_preparacion_datos', exist_ok=True)
    
    # Guardar datos completos
    df_features.to_csv('../03_preparacion_datos/datos_procesados.csv', index=False)
    print("\nOK Datos procesados guardados")
    
    # Guardar conjuntos de entrenamiento y prueba
    X_train.to_csv('../03_preparacion_datos/X_train.csv', index=False)
    X_test.to_csv('../03_preparacion_datos/X_test.csv', index=False)
    y_train.to_csv('../03_preparacion_datos/y_train.csv', index=False)
    y_test.to_csv('../03_preparacion_datos/y_test.csv', index=False)
    
    print("OK Conjuntos de entrenamiento y prueba guardados")
    print("\n" + "="*70)
    print("PREPARACIÓN COMPLETADA")
    print("="*70)

if __name__ == "__main__":
    def clasificar_riesgo(promedio):
        if promedio < 3.5:
            return 'alto'
        elif promedio < 4.0:
            return 'medio'
        else:
            return 'bajo'
    
    main()

