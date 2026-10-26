from typing import List
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

# Оставил только один класс (в вашем коде их было два)
class IrisRequestData(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Длина чашелистика в см", example=5.1)
    sepal_width: float = Field(..., gt=0, description="Ширина чашелистика в см", example=3.5)
    petal_length: float = Field(..., gt=0, description="Длина лепестка в см", example=1.4)
    petal_width: float = Field(..., gt=0, description="Ширина лепестка в см", example=0.2)

class IrisResponseData(BaseModel):
    predicted_class: int = Field(..., description="Индекс предсказанного класса (0, 1 или 2)")
    probabilities: List[float] = Field(..., description="Вектор вероятностей распределения по классам")
    model_status: str = Field(..., description="Системный статус и идентификатор версии инференса")

onnx_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global onnx_session
    print("Инициализация сервера: загрузка графа ONNX в оперативную память...")
    onnx_session = ort.InferenceSession("iris_model.onnx")
    yield
    print("Остановка сервера: освобождение ресурсов памяти...")
    del onnx_session

app = FastAPI(lifespan=lifespan)

@app.post("/v1/predict", response_model=IrisResponseData)
def predict(data: IrisRequestData):
    input_array = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]], dtype=np.float32)

    input_name = onnx_session.get_inputs()[0].name
    raw_predictions = onnx_session.run(None, {input_name: input_array})

    # ONNX Runtime возвращает список тензоров. Извлекаем нужные элементы для батча размера 1
    predicted_class = int(raw_predictions[0][0])

    # --- ИСПРАВЛЕНИЕ ОШИБКИ ---
    prob_output = raw_predictions[1][0]

    if isinstance(prob_output, dict):
        # Если ONNX вернул словарь (ZipMap), извлекаем только значения
        probabilities = list(prob_output.values())
    else:
        # Если вернулся классический numpy-массив, используем tolist()
        probabilities = prob_output.tolist()
    # --------------------------

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "model_status": "iris_onnx_v1.0"
    }
