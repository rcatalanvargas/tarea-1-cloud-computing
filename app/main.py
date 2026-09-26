import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------
# Rutas y carga de los artefactos
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
METADATA_PATH = PROJECT_ROOT / "model" / "metadata.json"

try:
    modelo = joblib.load(MODEL_PATH)

    with METADATA_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as archivo:
        metadata = json.load(archivo)

except FileNotFoundError as error:
    raise RuntimeError(
        "No se encontraron los artefactos del modelo."
    ) from error


MODEL_CLASSES = [
    str(clase)
    for clase in modelo
    .named_steps["clasificador"]
    .classes_
]

if "Placed" not in MODEL_CLASSES:
    raise RuntimeError(
        "La clase 'Placed' no está disponible en el modelo."
    )


# ---------------------------------------------------------
# Contratos informativos
# ---------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    loader: bool

class ModelInfoResponse(BaseModel):
    estimator_name: str
    target: str
    classes: list[str]
    input_feature_count: int
    input_features: list[str]
    scikit_learn_version: str
    test_metrics: dict[str, float]


# ---------------------------------------------------------
# Contrato de entrada
# ---------------------------------------------------------

class StudentInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Age": 22,
                "Gender": "Female",
                "University_Year": "Senior",
                "Major": "Data Science",
                "Attendance_Percentage": 85,
                "Study_Hours_Per_Week": 25,
                "CGPA": 3.4,
                "Programming_Skill": 8,
                "Projects_Completed": 10,
                "Certifications": 3,
                "Hackathons": 4,
                "GitHub_Profile": "Yes",
                "Internships": 4,
                "Leadership_Experience": "Yes",
                "LinkedIn_Profile": "Yes",
                "Resume_Score": 95,
                "Communication_Skills": 8,
                "Teamwork": 8,
                "Problem_Solving": 9,
                "English_Proficiency": "Advanced",
                "Interview_Score": 88,
            }
        },
    )

    Age: int = Field(ge=18, le=30)
    Gender: Literal["Female", "Male", "Other"]

    University_Year: Literal[
        "Freshman",
        "Junior",
        "Senior",
        "Sophomore",
    ]

    Major: Literal[
        "Artificial Intelligence",
        "Business Analytics",
        "Computer Science",
        "Cybersecurity",
        "Data Science",
        "Electrical Engineering",
        "Information Technology",
        "Software Engineering",
    ]

    Attendance_Percentage: int = Field(ge=50, le=100)
    Study_Hours_Per_Week: int = Field(ge=5, le=45)
    CGPA: float = Field(ge=2.0, le=4.0)
    Programming_Skill: int = Field(ge=1, le=10)
    Projects_Completed: int = Field(ge=0, le=15)
    Certifications: int = Field(ge=0, le=8)
    Hackathons: int = Field(ge=0, le=10)

    GitHub_Profile: Literal["No", "Yes"]

    Internships: int = Field(ge=0, le=5)

    Leadership_Experience: Literal["No", "Yes"]
    LinkedIn_Profile: Literal["No", "Yes"]

    Resume_Score: int = Field(ge=44, le=100)
    Communication_Skills: int = Field(ge=3, le=10)
    Teamwork: int = Field(ge=2, le=10)
    Problem_Solving: int = Field(ge=1, le=10)

    English_Proficiency: Literal[
        "Advanced",
        "Basic",
        "Intermediate",
    ]

    Interview_Score: int = Field(ge=17, le=100)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    students: list[StudentInput] = Field(
        min_length=1,
        max_length=100,
    )


# ---------------------------------------------------------
# Contratos de salida
# ---------------------------------------------------------

class PredictionResponse(BaseModel):
    prediction: Literal["Not Placed", "Placed"]
    probability_placed: float = Field(ge=0.0, le=1.0)
    class_probabilities: dict[str, float]
    version: str


class BatchPredictionResponse(BaseModel):
    prediction_count: int = Field(ge=1)
    predictions: list[PredictionResponse]


# ---------------------------------------------------------
# Función interna compartida
# ---------------------------------------------------------

def build_prediction_response(
    prediction: str,
    probabilities,
) -> PredictionResponse:
    class_probabilities = {
        clase: round(float(probability), 6)
        for clase, probability in zip(
            MODEL_CLASSES,
            probabilities,
        )
    }

    return PredictionResponse(
        prediction=prediction,
        probability_placed=(
            class_probabilities["Placed"]
        ),
        class_probabilities=class_probabilities,
        version="1.0.0",
    )


# ---------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="Student Career Success API",
    description=(
        "API para predecir el estado de colocación laboral "
        "de uno o varios estudiantes."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# Endpoints informativos
# ---------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health() -> HealthResponse:
    return HealthResponse(status="ok",loader=modelo is not None,)


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        estimator_name=metadata["estimator"]["name"],
        target=metadata["target"],
        classes=metadata["classes"],
        input_feature_count=len(
            metadata["input_features"]
        ),
        input_features=metadata["input_features"],
        scikit_learn_version=(
            metadata["versions"]["scikit-learn"]
        ),
        test_metrics=metadata["metrics"]["test"],
    )


# ---------------------------------------------------------
# Predicción individual
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(
    student: StudentInput,
) -> PredictionResponse:
    try:
        student_df = pd.DataFrame(
            [student.model_dump()]
        )

        student_df = student_df[
            metadata["input_features"]
        ]

        prediction = str(
            modelo.predict(student_df)[0]
        )

        probabilities = modelo.predict_proba(
            student_df
        )[0]

        return build_prediction_response(
            prediction=prediction,
            probabilities=probabilities,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "No fue posible realizar la predicción."
            ),
        ) from error


# ---------------------------------------------------------
# Predicción por lote
# ---------------------------------------------------------

@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
def predict_batch(
    batch: BatchPredictionRequest,
) -> BatchPredictionResponse:
    try:
        students_df = pd.DataFrame(
            [
                student.model_dump()
                for student in batch.students
            ]
        )

        students_df = students_df[
            metadata["input_features"]
        ]

        predictions = modelo.predict(students_df)
        probabilities = modelo.predict_proba(
            students_df
        )

        prediction_responses = [
            build_prediction_response(
                prediction=str(prediction),
                probabilities=student_probabilities,
            )
            for prediction, student_probabilities in zip(
                predictions,
                probabilities,
            )
        ]

        return BatchPredictionResponse(
            prediction_count=len(
                prediction_responses
            ),
            predictions=prediction_responses,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "No fue posible realizar "
                "las predicciones por lote."
            ),
        ) from error