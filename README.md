# Modelo Predictivo de Riesgo de Repitencia - CRISP-DM

Este proyecto implementa un modelo predictivo para clasificar el riesgo de repitencia de estudiantes chilenos basado en el avance de sus notas, este modelo fue entrenado con los datos de los alumnos egresados de la enseñanza media en el año 2024, estos son datos abiertos que Mineduc sube año a año, sin embargo en dichos csv solo existen los promedios de los estudiantes, no sus notas como tal, es por este motivo que las notas de los estudiantes han sido simuladas a partir de los promedios existentes.

## Estructura del Proyecto
```
├── 01_comprension_negocio/
│   └── objetivos.md
├── 02_comprension_datos/
│   ├── eda.py
│   └── reporte_eda.html
├── 03_preparacion_datos/
│   ├── preparacion.py
│   └── datos_procesados.csv
├── 04_modelado/
│   ├── entrenamiento.py
│   └── modelo_entrenado.pkl
├── 05_evaluacion/
│   ├── evaluacion.py
│   └── metricas.txt
├── 06_despliegue/
│   ├── app.py
│   └── modelo_final.pkl
└── requirements.txt
```

## Metodología CRISP-DM

### Fase 1: Comprensión del Negocio
- **Objetivo**: Predecir riesgo de repitencia basado en promedios de notas
- **Criterio**: Promedio >= 4.0 pasa, < 4.0 repite
- **Clasificación**: Alto, Medio, Bajo riesgo

### Fase 2: Comprensión de los Datos
- Análisis exploratorio del dataset
- Identificación de variables relevantes
- Análisis de distribución de promedios

### Fase 3: Preparación de Datos
- Limpieza y transformación
- Feature engineering
- Creación de variable objetivo (riesgo)

### Fase 4: Modelado
- Entrenamiento de modelos de clasificación
- Optimización de hiperparámetros

### Fase 5: Evaluación
- Validación cruzada
- Métricas de rendimiento
- Selección del mejor modelo

### Fase 6: Despliegue
- API REST para predicciones
- Recibe hasta 3 notas
- Calcula promedio y clasifica riesgo

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

1. Ejecutar análisis exploratorio:
```bash
python 02_comprension_datos/eda.py
```

2. Preparar datos:
```bash
python 03_preparacion_datos/preparacion.py
```

3. Entrenar modelo:
```bash
python 04_modelado/entrenamiento.py
```

4. Evaluar modelo:
```bash
python 05_evaluacion/evaluacion.py
```

5. Iniciar API:
```bash
python 06_despliegue/app.py
```

## API de Predicción

### Endpoint: POST /predict

## Ejemplo de entrada en Postman

**Request:**
```json
{
  "notas": [2.0, 7.0, 5.5]
}
```
## Salida
**Response:**
```json
{
  "promedio": 4.83,
  "riesgo": "bajo",
  "probabilidades": {
    "alto": 0.05,
    "medio": 0.15,
    "bajo": 0.80
  }
}
```




