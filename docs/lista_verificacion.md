# Lista de verificación final

Fecha de revisión: 26 de septiembre de 2026.

Esta lista registra el estado verificable de los requisitos de la Tarea Final de Cloud Computing.

## Dataset y problema

- [x] Se utiliza una tarea de clasificación supervisada.
- [x] La variable objetivo es `Placement_Status`.
- [x] El dataset contiene 50.000 filas, superando el mínimo de 500.
- [x] El modelo utiliza 21 variables predictoras, superando el mínimo de 4.
- [x] Se incluyen 7 variables categóricas y 14 numéricas.
- [x] La fuente pública de Kaggle está documentada en el README.
- [x] El dataset crudo está excluido del repositorio mediante `.gitignore`.
- [x] Las variables excluidas y los posibles riesgos de fuga de información están justificados.
- [x] Se identificó un desbalance moderado: 78,08 % `Placed` y 21,92 % `Not Placed`.

## E1 — Entrenamiento reproducible

- [x] El entrenamiento está documentado en `notebooks/00_Exploracion_inicial.ipynb`.
- [x] Se utiliza una separación entrenamiento/prueba de 80/20.
- [x] La separación es estratificada.
- [x] Se utiliza una semilla fija `random_state=42`.
- [x] Se aplica validación cruzada con `StratifiedKFold`, 5 particiones y semilla fija.
- [x] Se reportan `accuracy`, `balanced_accuracy`, `f1_macro` y `roc_auc`.
- [x] Las métricas se justifican considerando el desbalance de clases.

## E2 — Serialización del pipeline

- [x] El artefacto se encuentra en `model/model.pkl`.
- [x] El archivo contiene el pipeline completo: preprocesamiento y clasificador.
- [x] El pipeline incluye `StandardScaler`, `OneHotEncoder` y `LogisticRegression`.
- [x] El modelo fue cargado correctamente en un proceso distinto al entrenamiento.
- [x] El artefacto pesa aproximadamente 0,0069 MB, por debajo del límite de 100 MB.
- [x] `model/metadata.json` contiene versiones, variables ordenadas y métricas.

## E3 — Dependencias

- [x] Existe `requirements.txt`.
- [x] Las dependencias están declaradas con versiones fijas.
- [x] La versión de scikit-learn coincide con la utilizada para generar el modelo.
- [x] `python -m pip check` no reporta dependencias rotas.

## E4 — Intérprete

- [x] Existe `runtime.txt`.
- [x] Se declara Python 3.12.12.

## E5 — Comando de arranque

- [x] Existe `Procfile`.
- [x] El comando inicia FastAPI mediante Uvicorn.
- [x] Existe un `Dockerfile` para el despliegue en Google Cloud Run.
- [x] `.dockerignore` excluye el entorno virtual, pruebas, notebooks y archivos temporales.

## E6 — API de inferencia

- [x] La API está implementada en `app/main.py`.
- [x] FastAPI genera documentación automática en `/docs`.
- [x] El contrato de entrada y salida está validado con Pydantic.
- [x] Existe manejo de errores mediante respuestas HTTP.
- [x] Está disponible el endpoint `/health`.
- [x] Está disponible el endpoint `/model-info`.
- [x] Está disponible el endpoint `/predict`.
- [x] Está disponible el endpoint `/predict-batch`.
- [x] Una edad fuera del rango permitido produce una respuesta HTTP 422.

## E7 — Pruebas y evidencia local

- [x] Las pruebas están implementadas en `tests/test_api.py`.
- [x] Se ejecutaron 5 pruebas automatizadas correctamente.
- [x] La salida de las pruebas está guardada en `docs/resultado_pruebas.txt`.
- [x] La API fue ejecutada localmente con Uvicorn.
- [x] Swagger fue comprobado localmente en `/docs`.
- [x] Incorporar evidencia visual final de Swagger, una predicción válida y un error HTTP 422.
- [x] Repetir las pruebas después de todos los cambios finales y actualizar la evidencia.

## E8 — Repositorio GitHub

- [x] El repositorio tiene historial de commits.
- [x] El historial conserva la contribución de ambos integrantes.
- [x] El repositorio contiene código, modelo, metadatos, pruebas y documentación.
- [x] El README incluye instrucciones de reproducción.
- [ ] Confirmar el estado limpio del repositorio después del commit final.
- [ ] Integrar la rama de revisión en `main`.

## Despliegue opcional en la nube

- [x] La API está desplegada en Google Cloud Run.
- [x] La URL pública de Swagger está documentada en el README.
- [x] Los endpoints públicos fueron comprobados.
- [x] El repositorio está conectado con un flujo de despliegue continuo.
- [x] Existe evidencia del despliegue en `docs/Evidencias despliegue continuo.png`.
- [ ] Verificar nuevamente el servicio público después de integrar los cambios finales.

## Resultado de métricas en prueba

- `accuracy`: 0,7414.
- `balanced_accuracy`: 0,7296.
- `f1_macro`: 0,6825.
- `roc_auc`: 0,8031.
