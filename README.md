# Predicción de obtención de empleo de estudiantes

## Descripción del proyecto

Este proyecto implementa un sistema de clasificación supervisada para predecir si un estudiante obtuvo empleo.

El trabajo incluye la exploración de datos, el entrenamiento y evaluación de un modelo, la serialización del pipeline completo y la implementación de una API de inferencia con FastAPI.

## Objetivo

Predecir la variable `Placement_Status`, cuyas clases son:

- `Placed`: estudiante que obtuvo empleo.
- `Not Placed`: estudiante que no obtuvo empleo.

## Dataset

Se utiliza el archivo `student_career_success_dataset.csv`, que contiene:

- 50.000 registros.
- 29 columnas originales.
- Variables numéricas y categóricas.
- Sin valores nulos.
- Sin registros duplicados.

**Fuente pública:** [Student Career Success Prediction Dataset — Mobeen Fatima, Kaggle](https://www.kaggle.com/datasets/mobeenfatimah/student-career-success-prediction-dataset)

## Metodología

El problema se abordó como una clasificación binaria supervisada. El momento definido para realizar la predicción es después de la entrevista y antes de conocer el resultado laboral del estudiante.

### Variables excluidas

Se excluyeron las siguientes columnas:

- `Student_ID`: identificador único sin valor predictivo generalizable.
- `Academic_Performance`: información redundante derivada del `CGPA`.
- `Company_Tier`: información conocida después de obtener empleo.
- `Career_Field`: resultado profesional posterior.
- `Placement_Mode`: revela directamente la forma de obtención del empleo.
- `Starting_Salary_USD`: información disponible después de obtener empleo.
- `Employability_Score`: puntuación compuesta excluida después de comprobar que no producía una mejora relevante en las métricas.

Las variables relacionadas con resultados posteriores fueron excluidas para evitar fuga de información entre la variable objetivo y los predictores.

### Preparación de los datos

Se utilizaron 21 variables predictoras:

- 14 variables numéricas.
- 7 variables categóricas.

Los datos se dividieron de la siguiente manera:

- 80 % para entrenamiento: 40.000 registros.
- 20 % para prueba: 10.000 registros.
- Semilla fija: `random_state=42`.
- División estratificada para conservar la proporción de las clases.

También se aplicó validación cruzada estratificada de 5 particiones sobre los datos de entrenamiento.

### Pipeline del modelo

El pipeline completo contiene:

1. `StandardScaler` para las variables numéricas.
2. `OneHotEncoder(handle_unknown="ignore")` para las variables categóricas.
3. `LogisticRegression` con `class_weight="balanced"` como clasificador.

El preprocesamiento y el clasificador se almacenan juntos en `model/model.pkl`, evitando diferencias entre el entrenamiento y la inferencia.

## Evaluación del modelo

La variable objetivo presenta un desbalance moderado:

- `Placed`: 78,08 %.
- `Not Placed`: 21,92 %.

Por esta razón, la exactitud (`accuracy`) no se utilizó de manera aislada. Se reportaron las siguientes métricas:

- `accuracy`: proporción total de predicciones correctas.
- `balanced_accuracy`: entrega la misma importancia a ambas clases.
- `f1_macro`: calcula el F1 de cada clase y luego los promedia sin favorecer a la clase mayoritaria.
- `roc_auc`: evalúa la capacidad del modelo para separar ambas clases utilizando probabilidades.

### Validación cruzada sobre entrenamiento

Promedios obtenidos mediante validación cruzada estratificada de 5 particiones:

| Métrica | Media |
|---|---:|
| Accuracy | 0.7485 |
| Balanced accuracy | 0.7380 |
| F1 macro | 0.6904 |
| ROC-AUC | 0.8114 |

### Evaluación final sobre prueba

El conjunto de prueba se mantuvo separado durante la selección del modelo y se utilizó para la evaluación final:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7414 |
| Balanced accuracy | 0.7296 |
| F1 macro | 0.6825 |
| ROC-AUC | 0.8031 |

Debido al desbalance, las métricas principales para interpretar el modelo son `balanced_accuracy`, `f1_macro` y `roc_auc`. La similitud entre los resultados de validación cruzada y prueba indica un comportamiento estable fuera de los datos utilizados para entrenar.

## Instalación y ejecución local

### 1. Crear el entorno virtual

Desde la raíz del proyecto:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar las dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Ejecutar la API

```bash
python -m uvicorn app.main:app --reload
```

El servicio quedará disponible en:

- API: <http://127.0.0.1:8000>
- Documentación Swagger: <http://127.0.0.1:8000/docs>
- Estado del servicio: <http://127.0.0.1:8000/health>
- Información del modelo: <http://127.0.0.1:8000/model-info>

Para detener el servicio se utiliza `Control + C` en la terminal donde se está ejecutando.

## Pruebas automatizadas

Las pruebas se ejecutan desde la raíz del proyecto con:

```bash
python -m pytest -v
```

La suite verifica:

- Estado general de la API.
- Información del modelo.
- Predicción individual válida.
- Rechazo de una edad fuera del rango permitido.
- Predicción de múltiples estudiantes.

Resultado obtenido:

```text
5 passed
```

La evidencia completa de la ejecución se encuentra en [`docs/resultado_pruebas.txt`](docs/resultado_pruebas.txt).

## API de inferencia

La API fue implementada con FastAPI y valida automáticamente los datos de entrada mediante modelos de Pydantic.

### Endpoints

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/health` | Verifica que el servicio y el modelo estén disponibles. |
| `GET` | `/model-info` | Entrega información del modelo, clases y métricas. |
| `POST` | `/predict` | Realiza una predicción individual. |
| `POST` | `/predict-batch` | Realiza entre 1 y 100 predicciones en una solicitud. |
| `GET` | `/docs` | Muestra la documentación interactiva de Swagger. |

### Ejemplo de entrada para `/predict`

```json
{
  "Age": 22,
  "Gender": "Male",
  "University_Year": "Senior",
  "Major": "Data Science",
  "Attendance_Percentage": 90,
  "Study_Hours_Per_Week": 25,
  "CGPA": 3.5,
  "Programming_Skill": 8,
  "Projects_Completed": 10,
  "Certifications": 3,
  "Hackathons": 4,
  "GitHub_Profile": "Yes",
  "Internships": 4,
  "Leadership_Experience": "Yes",
  "LinkedIn_Profile": "Yes",
  "Resume_Score": 95,
  "Communication_Skills": 9,
  "Teamwork": 8,
  "Problem_Solving": 9,
  "English_Proficiency": "Advanced",
  "Interview_Score": 90
}
```

La respuesta informa:

- Clase predicha.
- Probabilidad de obtener empleo.
- Probabilidad estimada para cada clase.

Si los datos no cumplen el contrato —por ejemplo, una edad fuera del rango permitido o una categoría inválida— la API responde con el código HTTP `422` y el detalle de validación.

## Estructura del proyecto

```text
Tarea_Final_Cloud_Computing/
├── app/
│   ├── __init__.py
│   └── main.py
├── docs/
│   └── resultado_pruebas.txt
├── model/
│   ├── metadata.json
│   └── model.pkl
├── notebooks/
│   └── 00_Exploracion_inicial.ipynb
├── tests/
│   └── test_api.py
├── .gitignore
├── Procfile
├── README.md
├── requirements.txt
└── runtime.txt
```

El dataset crudo y los entornos virtuales se mantienen fuera del repositorio mediante `.gitignore`.

Para reproducir el entrenamiento, el archivo `student_career_success_dataset.csv` debe descargarse desde la fuente de Kaggle indicada anteriormente y ubicarse en la raíz del proyecto.
