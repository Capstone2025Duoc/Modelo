"""
API de Despliegue
Fase 6 de CRISP-DM: Despliegue
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas

# Cargar modelo al iniciar
print("Cargando modelo...")
try:
    # Obtener el directorio base del proyecto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    modelo_path = os.path.join(base_dir, '04_modelado', 'modelo_riesgo_repitencia.pkl')
    modelo = joblib.load(modelo_path)
    print("OK Modelo cargado exitosamente")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    modelo = None

def clasificar_riesgo(promedio):
    """Clasifica el riesgo basado en el promedio"""
    if promedio < 3.5:
        return 'alto'
    elif promedio < 4.0:
        return 'medio'
    else:
        return 'bajo'

def calcular_features(notas):
    """
    Calcula las features necesarias para el modelo
    a partir de las notas ingresadas
    """
    # Asegurar que tenemos máximo 3 notas
    notas = notas[:3]
    
    # Rellenar con NaN si hay menos de 3 notas
    while len(notas) < 3:
        notas.append(np.nan)
    
    nota_1, nota_2, nota_3 = notas[0], notas[1], notas[2]
    
    # Calcular promedio
    notas_validas = [n for n in notas if not pd.isna(n)]
    promedio_calculado = np.mean(notas_validas) if notas_validas else 0
    
    # Cantidad de notas
    cantidad_notas = len(notas_validas)
    
    # Tendencia
    if len(notas_validas) >= 2:
        if notas_validas[-1] > notas_validas[0]:
            tendencia = 1  # Mejora
        elif notas_validas[-1] < notas_validas[0]:
            tendencia = -1  # Empeora
        else:
            tendencia = 0  # Estable
    else:
        tendencia = 0
    
    # Variabilidad
    if len(notas_validas) >= 2:
        variabilidad = np.std(notas_validas)
    else:
        variabilidad = 0
    
    # Nota mínima y máxima
    nota_min = min(notas_validas) if notas_validas else 0
    nota_max = max(notas_validas) if notas_validas else 0
    
    # Rellenar NaN con 0 para el modelo
    nota_1 = nota_1 if not pd.isna(nota_1) else 0
    nota_2 = nota_2 if not pd.isna(nota_2) else 0
    nota_3 = nota_3 if not pd.isna(nota_3) else 0
    
    # Features para el modelo (SIN promedio_calculado ni distancia_umbral para evitar data leakage)
    features = {
        'nota_1': nota_1,
        'nota_2': nota_2,
        'nota_3': nota_3,
        'cantidad_notas': cantidad_notas,
        'tendencia': tendencia,
        'variabilidad': variabilidad,
        'nota_min': nota_min,
        'nota_max': nota_max
    }
    
    # Guardar promedio_calculado para mostrarlo en la respuesta (pero no como feature)
    features['_promedio_calculado'] = promedio_calculado  # Prefijo _ para indicar que no es feature del modelo
    
    return features

@app.route('/')
def home():
    """Sirve el formulario HTML"""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/api')
def api_info():
    """Endpoint de información de la API"""
    return jsonify({
        'mensaje': 'API de Predicción de Riesgo de Repitencia',
        'version': '1.0',
        'endpoints': {
            '/': 'Formulario web',
            '/api': 'Información de la API',
            '/predict': 'POST - Predicción de riesgo',
            '/health': 'GET - Estado del servicio'
        }
    })

@app.route('/health')
def health():
    """Endpoint de salud del servicio"""
    return jsonify({
        'status': 'healthy',
        'modelo_cargado': modelo is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de predicción
    
    Request body:
    {
        "notas": [2.0, 7.0, 5.5]  # Hasta 3 notas
    }
    
    Response:
    {
        "promedio": 4.83,
        "riesgo": "bajo",
        "probabilidades": {
            "alto": 0.05,
            "medio": 0.15,
            "bajo": 0.80
        },
        "features": {...}
    }
    """
    if modelo is None:
        return jsonify({
            'error': 'Modelo no disponible'
        }), 500
    
    try:
        # Obtener datos del request
        data = request.get_json()
        
        if not data or 'notas' not in data:
            return jsonify({
                'error': 'Se requiere el campo "notas" en el body'
            }), 400
        
        notas = data['notas']
        
        # Validar que sea una lista
        if not isinstance(notas, list):
            return jsonify({
                'error': 'El campo "notas" debe ser una lista'
            }), 400
        
        # Validar que tenga máximo 3 notas
        if len(notas) > 3:
            return jsonify({
                'error': 'Se pueden ingresar máximo 3 notas'
            }), 400
        
        # Validar que las notas estén en el rango válido
        for nota in notas:
            if not isinstance(nota, (int, float)):
                return jsonify({
                    'error': f'La nota {nota} no es un número válido'
                }), 400
            if nota < 1.0 or nota > 7.0:
                return jsonify({
                    'error': f'La nota {nota} está fuera del rango válido (1.0 - 7.0)'
                }), 400
        
        # Calcular features
        features = calcular_features(notas)
        
        # Preparar datos para el modelo (excluir _promedio_calculado que no es feature)
        features_modelo = {k: v for k, v in features.items() if not k.startswith('_')}
        X = pd.DataFrame([features_modelo])
        
        # Realizar predicción
        prediccion = modelo.predict(X)[0]
        
        # Obtener probabilidades
        probabilidades = modelo.predict_proba(X)[0]
        clases = modelo.classes_
        
        # Crear diccionario de probabilidades
        prob_dict = {
            clase: float(prob) 
            for clase, prob in zip(clases, probabilidades)
        }
        
        # Calcular promedio (para mostrar en respuesta, pero no usado por el modelo)
        promedio = features['_promedio_calculado']
        
        # Respuesta
        respuesta = {
            'promedio': round(promedio, 2),
            'riesgo': prediccion,
            'probabilidades': prob_dict,
            'cantidad_notas': int(features['cantidad_notas']),
            'tendencia': 'mejora' if features['tendencia'] > 0 else 'empeora' if features['tendencia'] < 0 else 'estable'
        }
        
        return jsonify(respuesta)
    
    except Exception as e:
        return jsonify({
            'error': f'Error al procesar la solicitud: {str(e)}'
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Endpoint para predicciones en lote
    
    Request body:
    {
        "estudiantes": [
            {"id": 1, "notas": [2.0, 7.0]},
            {"id": 2, "notas": [3.0, 3.5, 3.8]}
        ]
    }
    """
    if modelo is None:
        return jsonify({
            'error': 'Modelo no disponible'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'estudiantes' not in data:
            return jsonify({
                'error': 'Se requiere el campo "estudiantes" en el body'
            }), 400
        
        estudiantes = data['estudiantes']
        resultados = []
        
        for estudiante in estudiantes:
            notas = estudiante.get('notas', [])
            estudiante_id = estudiante.get('id', None)
            
            # Calcular features
            features = calcular_features(notas)
            
            # Preparar datos para el modelo (excluir _promedio_calculado)
            features_modelo = {k: v for k, v in features.items() if not k.startswith('_')}
            X = pd.DataFrame([features_modelo])
            
            # Predicción
            prediccion = modelo.predict(X)[0]
            probabilidades = modelo.predict_proba(X)[0]
            clases = modelo.classes_
            
            prob_dict = {
                clase: float(prob) 
                for clase, prob in zip(clases, probabilidades)
            }
            
            resultados.append({
                'id': estudiante_id,
                'promedio': round(features['_promedio_calculado'], 2),
                'riesgo': prediccion,
                'probabilidades': prob_dict
            })
        
        return jsonify({
            'resultados': resultados,
            'total': len(resultados)
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Error al procesar el lote: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*70}")
    print("API de Predicción de Riesgo de Repitencia")
    print("="*70)
    print(f"Servidor iniciado en http://localhost:{port}")
    print(f"Modelo cargado: {modelo is not None}")
    print("\nEndpoints disponibles:")
    print(f"  GET  http://localhost:{port}/")
    print(f"  GET  http://localhost:{port}/health")
    print(f"  POST http://localhost:{port}/predict")
    print(f"  POST http://localhost:{port}/predict/batch")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)

