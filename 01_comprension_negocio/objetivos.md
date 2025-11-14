# Fase 1: Comprensión del Negocio

## Objetivos del Proyecto

### Objetivo Principal
Desarrollar un modelo predictivo que clasifique el riesgo de repitencia de estudiantes de los colegios en Chile basándose en sus notas.

### Objetivos Específicos
1. Predecir el riesgo de repitencia (alto, medio, bajo) basado en hasta 3 notas parciales
2. Considerar la evolución de las notas del estudiante
3. Proporcionar una herramienta de apoyo para la toma de decisiones educativas

## Criterios de Negocio

### Regla de Aprobación
- **Promedio >= 4.0**: El estudiante pasa de curso
- **Promedio < 4.0**: El estudiante repite

### Clasificación de Riesgo
- **Alto Riesgo**: Probabilidad alta de repitencia (promedio proyectado < 4.0)
- **Medio Riesgo**: Probabilidad moderada de repitencia
- **Bajo Riesgo**: Probabilidad baja de repitencia (promedio proyectado >= 4.0)

## Requisitos del Modelo

1. **Entrada**: Hasta 3 notas parciales (rango 1.0 - 7.0)
2. **Procesamiento**: 
   - Calcular promedio de las notas ingresadas
   - Considerar tendencia/evolución de las notas
3. **Salida**: 
   - Clasificación de riesgo (alto, medio, bajo)
   - Probabilidades asociadas
   - Promedio calculado

