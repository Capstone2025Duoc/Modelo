"""
Script de prueba para la API de predicción
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Prueba el endpoint de salud"""
    print("="*50)
    print("Test: Health Check")
    print("="*50)
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict_1_nota():
    """Prueba predicción con 1 nota"""
    print("="*50)
    print("Test: Predicción con 1 nota (2.0)")
    print("="*50)
    data = {
        "notas": [2.0]
    }
    response = requests.post(f"{API_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict_2_notas():
    """Prueba predicción con 2 notas"""
    print("="*50)
    print("Test: Predicción con 2 notas (2.0, 7.0)")
    print("="*50)
    data = {
        "notas": [2.0, 7.0]
    }
    response = requests.post(f"{API_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict_3_notas():
    """Prueba predicción con 3 notas"""
    print("="*50)
    print("Test: Predicción con 3 notas (2.0, 7.0, 5.5)")
    print("="*50)
    data = {
        "notas": [2.0, 7.0, 5.5]
    }
    response = requests.post(f"{API_URL}/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict_casos_varios():
    """Prueba varios casos"""
    print("="*50)
    print("Test: Varios casos de prueba")
    print("="*50)
    
    casos = [
        {"nombre": "Alto riesgo (1 nota baja)", "notas": [2.0]},
        {"nombre": "Alto riesgo (2 notas bajas)", "notas": [3.0, 3.5]},
        {"nombre": "Medio riesgo", "notas": [3.5, 3.8]},
        {"nombre": "Bajo riesgo (1 nota alta)", "notas": [6.0]},
        {"nombre": "Bajo riesgo (2 notas)", "notas": [5.0, 5.5]},
        {"nombre": "Bajo riesgo (3 notas)", "notas": [5.0, 6.0, 6.5]},
        {"nombre": "Mejora progresiva", "notas": [2.0, 4.0, 6.0]},
        {"nombre": "Empeora", "notas": [6.0, 4.0, 2.0]},
    ]
    
    for caso in casos:
        print(f"\nCaso: {caso['nombre']}")
        print(f"Notas: {caso['notas']}")
        response = requests.post(f"{API_URL}/predict", json={"notas": caso['notas']})
        if response.status_code == 200:
            result = response.json()
            print(f"  Promedio: {result['promedio']}")
            print(f"  Riesgo: {result['riesgo']}")
            print(f"  Tendencia: {result['tendencia']}")
        else:
            print(f"  Error: {response.json()}")
    print()

def test_batch():
    """Prueba predicción en lote"""
    print("="*50)
    print("Test: Predicción en lote")
    print("="*50)
    data = {
        "estudiantes": [
            {"id": 1, "notas": [2.0, 7.0]},
            {"id": 2, "notas": [3.0, 3.5, 3.8]},
            {"id": 3, "notas": [5.0, 6.0, 6.5]},
            {"id": 4, "notas": [4.5]}
        ]
    }
    response = requests.post(f"{API_URL}/predict/batch", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def main():
    """Ejecuta todos los tests"""
    print("="*70)
    print("PRUEBAS DE LA API DE PREDICCIÓN")
    print("="*70)
    print(f"\nAsegúrate de que la API esté corriendo en {API_URL}")
    print()
    
    try:
        test_health()
        test_predict_1_nota()
        test_predict_2_notas()
        test_predict_3_notas()
        test_predict_casos_varios()
        test_batch()
        
        print("="*70)
        print("TODAS LAS PRUEBAS COMPLETADAS")
        print("="*70)
    
    except requests.exceptions.ConnectionError:
        print("ERROR: No se pudo conectar a la API.")
        print(f"Asegúrate de que la API esté corriendo en {API_URL}")
        print("Ejecuta: python 06_despliegue/app.py")

if __name__ == "__main__":
    main()




