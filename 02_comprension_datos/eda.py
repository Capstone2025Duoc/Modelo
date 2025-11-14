"""
Análisis Exploratorio de Datos (EDA)
Fase 2 de CRISP-DM: Comprensión de los Datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def cargar_datos():
    """Carga el dataset CSV"""
    print("Cargando datos...")
    # El CSV usa punto y coma como separador y coma como decimal
    df = pd.read_csv(
        '../20230313_Notas_y_Egresados_Enseñanza_Media_2024_PUBL.csv',
        sep=';',
        decimal=',',
        encoding='utf-8'
    )
    print(f"Datos cargados: {len(df)} registros")
    return df

def exploracion_inicial(df):
    """Exploración inicial del dataset"""
    print("\n" + "="*50)
    print("EXPLORACIÓN INICIAL")
    print("="*50)
    
    print(f"\nDimensiones: {df.shape}")
    print(f"\nColumnas: {list(df.columns)}")
    print(f"\nTipos de datos:")
    print(df.dtypes)
    print(f"\nPrimeras filas:")
    print(df.head())
    print(f"\nInformación general:")
    print(df.info())
    print(f"\nEstadísticas descriptivas:")
    print(df.describe())
    
    return df

def analizar_promedios(df):
    """Análisis específico de la variable PROM_NOTAS_ALU"""
    print("\n" + "="*50)
    print("ANÁLISIS DE PROMEDIOS (PROM_NOTAS_ALU)")
    print("="*50)
    
    # Convertir a numérico si es necesario
    if df['PROM_NOTAS_ALU'].dtype == 'object':
        df['PROM_NOTAS_ALU'] = df['PROM_NOTAS_ALU'].str.replace(',', '.').astype(float)
    
    print(f"\nEstadísticas de promedios:")
    print(df['PROM_NOTAS_ALU'].describe())
    print(f"\nValores nulos: {df['PROM_NOTAS_ALU'].isnull().sum()}")
    print(f"\nRango: {df['PROM_NOTAS_ALU'].min()} - {df['PROM_NOTAS_ALU'].max()}")
    
    # Distribución
    print(f"\nDistribución de promedios:")
    print(f"< 4.0 (Riesgo repitencia): {(df['PROM_NOTAS_ALU'] < 4.0).sum()} ({(df['PROM_NOTAS_ALU'] < 4.0).sum()/len(df)*100:.2f}%)")
    print(f">= 4.0 (Aprobado): {(df['PROM_NOTAS_ALU'] >= 4.0).sum()} ({(df['PROM_NOTAS_ALU'] >= 4.0).sum()/len(df)*100:.2f}%)")
    
    return df

def analizar_egreso(df):
    """Análisis de la variable MARCA_EGRESO"""
    print("\n" + "="*50)
    print("ANÁLISIS DE EGRESO (MARCA_EGRESO)")
    print("="*50)
    
    print(f"\nDistribución:")
    print(df['MARCA_EGRESO'].value_counts())
    print(f"\nPorcentajes:")
    print(df['MARCA_EGRESO'].value_counts(normalize=True) * 100)
    
    # Relación entre promedio y egreso
    print(f"\nRelación Promedio vs Egreso:")
    relacion = df.groupby('MARCA_EGRESO')['PROM_NOTAS_ALU'].agg(['mean', 'std', 'count'])
    print(relacion)
    
    return df

def analizar_por_ano(df):
    """Análisis por año"""
    print("\n" + "="*50)
    print("ANÁLISIS POR AÑO")
    print("="*50)
    
    if 'AGNO' in df.columns:
        print(f"\nAños disponibles: {sorted(df['AGNO'].unique())}")
        print(f"\nDistribución por año:")
        print(df['AGNO'].value_counts().sort_index())
        
        print(f"\nPromedio por año:")
        promedio_ano = df.groupby('AGNO')['PROM_NOTAS_ALU'].agg(['mean', 'std', 'count'])
        print(promedio_ano)
    
    return df

def crear_visualizaciones(df):
    """Crea visualizaciones del EDA"""
    print("\n" + "="*50)
    print("CREANDO VISUALIZACIONES")
    print("="*50)
    
    # Crear directorio si no existe
    os.makedirs('../02_comprension_datos/graficos', exist_ok=True)
    
    # 1. Distribución de promedios
    plt.figure(figsize=(12, 6))
    plt.hist(df['PROM_NOTAS_ALU'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=4.0, color='r', linestyle='--', linewidth=2, label='Umbral de aprobación (4.0)')
    plt.xlabel('Promedio de Notas')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Promedios de Notas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../02_comprension_datos/graficos/distribucion_promedios.png', dpi=300)
    plt.close()
    print("OK Gráfico de distribución guardado")
    
    # 2. Boxplot de promedios por egreso
    plt.figure(figsize=(10, 6))
    df.boxplot(column='PROM_NOTAS_ALU', by='MARCA_EGRESO', ax=plt.gca())
    plt.axhline(y=4.0, color='r', linestyle='--', linewidth=2, label='Umbral de aprobación')
    plt.suptitle('')
    plt.title('Distribución de Promedios por Estado de Egreso')
    plt.xlabel('Egreso (1=Sí, 0=No)')
    plt.ylabel('Promedio de Notas')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../02_comprension_datos/graficos/boxplot_egreso.png', dpi=300)
    plt.close()
    print("OK Boxplot por egreso guardado")
    
    # 3. Distribución por año (si existe)
    if 'AGNO' in df.columns:
        plt.figure(figsize=(14, 6))
        promedio_ano = df.groupby('AGNO')['PROM_NOTAS_ALU'].mean()
        plt.plot(promedio_ano.index, promedio_ano.values, marker='o', linewidth=2, markersize=8)
        plt.axhline(y=4.0, color='r', linestyle='--', linewidth=2, label='Umbral de aprobación')
        plt.xlabel('Año')
        plt.ylabel('Promedio Medio')
        plt.title('Evolución del Promedio Medio por Año')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../02_comprension_datos/graficos/evolucion_ano.png', dpi=300)
        plt.close()
        print("OK Gráfico de evolución por año guardado")
    
    # 4. Distribución de riesgo (basado en promedio)
    plt.figure(figsize=(10, 6))
    df['RIESGO'] = pd.cut(df['PROM_NOTAS_ALU'], 
                         bins=[0, 3.0, 3.9, 4.0, 7.0],
                         labels=['Alto (<3.0)', 'Alto (3.0-3.9)', 'Bajo (4.0-4.9)', 'Bajo (>=5.0)'],
                         include_lowest=True)
    riesgo_counts = df['RIESGO'].value_counts()
    plt.bar(riesgo_counts.index, riesgo_counts.values, color=['red', 'orange', 'lightgreen', 'green'], alpha=0.7)
    plt.xlabel('Categoría de Riesgo')
    plt.ylabel('Cantidad de Estudiantes')
    plt.title('Distribución de Riesgo de Repitencia')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../02_comprension_datos/graficos/distribucion_riesgo.png', dpi=300)
    plt.close()
    print("OK Gráfico de distribución de riesgo guardado")
    
    # Eliminar columna temporal
    df = df.drop('RIESGO', axis=1)
    
    return df

def generar_reporte(df):
    """Genera un reporte resumen del EDA"""
    reporte = []
    reporte.append("="*70)
    reporte.append("REPORTE DE ANÁLISIS EXPLORATORIO DE DATOS")
    reporte.append("="*70)
    reporte.append(f"\nFecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append(f"\nTotal de registros: {len(df):,}")
    reporte.append(f"Total de columnas: {len(df.columns)}")
    reporte.append(f"\nValores nulos en PROM_NOTAS_ALU: {df['PROM_NOTAS_ALU'].isnull().sum()}")
    reporte.append(f"\nRango de promedios: {df['PROM_NOTAS_ALU'].min():.2f} - {df['PROM_NOTAS_ALU'].max():.2f}")
    reporte.append(f"Promedio general: {df['PROM_NOTAS_ALU'].mean():.2f}")
    reporte.append(f"Mediana: {df['PROM_NOTAS_ALU'].median():.2f}")
    reporte.append(f"Desviación estándar: {df['PROM_NOTAS_ALU'].std():.2f}")
    
    reporte.append(f"\nDistribución por umbral de aprobación:")
    reporte.append(f"  - Promedio < 4.0: {(df['PROM_NOTAS_ALU'] < 4.0).sum():,} ({(df['PROM_NOTAS_ALU'] < 4.0).sum()/len(df)*100:.2f}%)")
    reporte.append(f"  - Promedio >= 4.0: {(df['PROM_NOTAS_ALU'] >= 4.0).sum():,} ({(df['PROM_NOTAS_ALU'] >= 4.0).sum()/len(df)*100:.2f}%)")
    
    reporte.append(f"\nDistribución por egreso:")
    reporte.append(f"  - Egresados (1): {(df['MARCA_EGRESO'] == 1).sum():,} ({(df['MARCA_EGRESO'] == 1).sum()/len(df)*100:.2f}%)")
    reporte.append(f"  - No egresados (0): {(df['MARCA_EGRESO'] == 0).sum():,} ({(df['MARCA_EGRESO'] == 0).sum()/len(df)*100:.2f}%)")
    
    if 'AGNO' in df.columns:
        reporte.append(f"\nAños en el dataset: {df['AGNO'].min()} - {df['AGNO'].max()}")
        reporte.append(f"Años únicos: {len(df['AGNO'].unique())}")
    
    reporte.append("\n" + "="*70)
    reporte.append("CONCLUSIONES")
    reporte.append("="*70)
    reporte.append("1. El dataset contiene información histórica de promedios de estudiantes")
    reporte.append("2. El umbral de aprobación es 4.0")
    reporte.append("3. Se puede crear una variable objetivo basada en el promedio")
    reporte.append("4. El modelo debe considerar la evolución de las notas")
    
    reporte_texto = "\n".join(reporte)
    
    # Guardar reporte
    with open('../02_comprension_datos/reporte_eda.txt', 'w', encoding='utf-8') as f:
        f.write(reporte_texto)
    
    print("\n" + reporte_texto)
    print("\nOK Reporte guardado en reporte_eda.txt")

def main():
    """Función principal"""
    print("="*70)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("Fase 2 de CRISP-DM: Comprensión de los Datos")
    print("="*70)
    
    # Cargar datos
    df = cargar_datos()
    
    # Exploración inicial
    df = exploracion_inicial(df)
    
    # Análisis de promedios
    df = analizar_promedios(df)
    
    # Análisis de egreso
    df = analizar_egreso(df)
    
    # Análisis por año
    df = analizar_por_ano(df)
    
    # Crear visualizaciones
    df = crear_visualizaciones(df)
    
    # Generar reporte
    generar_reporte(df)
    
    # Guardar datos procesados para siguiente fase
    df.to_csv('../03_preparacion_datos/datos_eda.csv', index=False)
    print("\nOK Datos guardados para siguiente fase (datos_eda.csv)")

if __name__ == "__main__":
    main()

