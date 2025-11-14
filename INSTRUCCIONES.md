# Instrucciones de Uso del Modelo Predictivo

## Requisitos Previos

1. Python 3.8 o superior
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Flujo de Trabajo Completo

### Paso 1: Análisis Exploratorio de Datos (EDA)

Ejecuta el análisis exploratorio para entender los datos:

```bash
python 02_comprension_datos/eda.py
```

Esto generará:
- Gráficos en `02_comprension_datos/graficos/`
- Reporte en `02_comprension_datos/reporte_eda.txt`
- Datos procesados en `02_comprension_datos/datos_eda.csv`

### Paso 2: Preparación de Datos

Prepara los datos para el modelado:

```bash
python 03_preparacion_datos/preparacion.py
```

Esto generará:
- Datos procesados en `03_preparacion_datos/datos_procesados.csv`
- Conjuntos de entrenamiento y prueba (X_train, X_test, y_train, y_test)

### Paso 3: Entrenamiento del Modelo

Entrena el modelo de machine learning:

```bash
python 04_modelado/entrenamiento.py
```

Esto generará:
- Modelo entrenado en `04_modelado/modelo_riesgo_repitencia.pkl`
- Modelo sin optimizar en `04_modelado/modelo_sin_optimizar.pkl`

### Paso 4: Evaluación del Modelo

Evalúa el rendimiento del modelo:

```bash
python 05_evaluacion/evaluacion.py
```

Esto generará:
- Gráficos de evaluación en `05_evaluacion/graficos/`
- Reporte de evaluación en `05_evaluacion/reporte_evaluacion.txt`

### Paso 5: Despliegue de la API

Inicia el servidor de la API:

```bash
python 06_despliegue/app.py
```

La API estará disponible en `http://localhost:5000`

### Paso 6: Probar la API

En otra terminal, ejecuta los tests:

```bash
python 06_despliegue/test_api.py
```

O prueba manualmente con curl:

```bash
# Predicción con 1 nota
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"notas": [2.0]}'

# Predicción con 2 notas
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"notas": [2.0, 7.0]}'

# Predicción con 3 notas
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"notas": [2.0, 7.0, 5.5]}'
```

## Ejemplos de Uso de la API

### Ejemplo 1: Estudiante con 1 nota (Riesgo Alto)

**Request:**
```json
{
  "notas": [2.0]
}
```

**Response:**
```json
{
  "promedio": 2.0,
  "riesgo": "alto",
  "probabilidades": {
    "alto": 0.85,
    "medio": 0.10,
    "bajo": 0.05
  },
  "cantidad_notas": 1,
  "tendencia": "estable",
  "distancia_umbral": -2.0
}
```

### Ejemplo 2: Estudiante con 2 notas (Mejora)

**Request:**
```json
{
  "notas": [2.0, 7.0]
}
```

**Response:**
```json
{
  "promedio": 4.5,
  "riesgo": "bajo",
  "probabilidades": {
    "alto": 0.10,
    "medio": 0.20,
    "bajo": 0.70
  },
  "cantidad_notas": 2,
  "tendencia": "mejora",
  "distancia_umbral": 0.5
}
```

### Ejemplo 3: Estudiante con 3 notas

**Request:**
```json
{
  "notas": [2.0, 7.0, 5.5]
}
```

**Response:**
```json
{
  "promedio": 4.83,
  "riesgo": "bajo",
  "probabilidades": {
    "alto": 0.05,
    "medio": 0.15,
    "bajo": 0.80
  },
  "cantidad_notas": 3,
  "tendencia": "mejora",
  "distancia_umbral": 0.83
}
```

## Endpoints de la API

### GET `/`
Información sobre la API

### GET `/health`
Estado del servicio y verificación del modelo

### POST `/predict`
Predicción individual
- **Body**: `{"notas": [2.0, 7.0, 5.5]}`
- **Response**: Predicción con probabilidades

### POST `/predict/batch`
Predicción en lote
- **Body**: `{"estudiantes": [{"id": 1, "notas": [2.0, 7.0]}, ...]}`
- **Response**: Array de predicciones

## Notas Importantes

1. **Rango de notas**: Las notas deben estar entre 1.0 y 7.0
2. **Máximo de notas**: Se pueden ingresar hasta 3 notas
3. **Cálculo de promedio**: Se calcula automáticamente el promedio de las notas ingresadas
4. **Clasificación de riesgo**:
   - **Alto**: Promedio < 3.5
   - **Medio**: Promedio entre 3.5 y 3.9
   - **Bajo**: Promedio >= 4.0

## Solución de Problemas

### Error: "Modelo no disponible"
- Asegúrate de haber ejecutado `04_modelado/entrenamiento.py` primero
- Verifica que el archivo `04_modelado/modelo_riesgo_repitencia.pkl` existe

### Error: "No se pudo conectar a la API"
- Verifica que la API esté corriendo: `python 06_despliegue/app.py`
- Verifica que el puerto 5000 esté disponible

### Error en la preparación de datos
- Verifica que el archivo CSV original esté en la raíz del proyecto
- Asegúrate de tener permisos de lectura/escritura en los directorios

## Estructura de Carpetas

```
Modelo/
├── 01_comprension_negocio/
├── 02_comprension_datos/
│   ├── graficos/
│   └── reporte_eda.txt
├── 03_preparacion_datos/
│   └── datos_procesados.csv
├── 04_modelado/
│   └── modelo_riesgo_repitencia.pkl
├── 05_evaluacion/
│   ├── graficos/
│   └── reporte_evaluacion.txt
├── 06_despliegue/
│   ├── app.py
│   └── test_api.py
└── requirements.txt
```




