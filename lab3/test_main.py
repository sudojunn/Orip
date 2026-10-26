import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    # Инициализируем TestClient внутри контекстного менеджера lifespan
    with TestClient(app) as test_client:
        yield test_client

def test_predict_success(client):
    valid_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/v1/predict", json=valid_payload)

    assert response.status_code == 200
    response_json = response.json()
    # Проверяем наличие всех обязательных полей контракта IrisResponseData
    assert "predicted_class" in response_json
    assert "probabilities" in response_json
    assert "model_status" in response_json
    # Верифицируем типы данных в ответе
    assert isinstance(response_json["predicted_class"], int)
    assert isinstance(response_json["probabilities"], list)
    assert len(response_json["probabilities"]) == 3 # Модель классифицирует 3 класса Ирисов


def test_predict_negative_value_error(client):
    # Формируем битый пакет: длина чашелистика отрицательная (-5.1)
    invalid_payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    # Отправляем POST-запрос на инференс
    response = client.post("/v1/predict", json=invalid_payload)
    # Утверждаем, что сервер ОБЯЗАН вернуть ошибку валидации 422
    assert response.status_code == 422
    # Проверяем, что в сообщении об ошибке Pydantic явно указал причину
    assert "Input should be greater than 0" in response.text

def test_predict_type_error(client):
    # Формируем битый пакет: вместо числа 0.2 передаем строку
    "string_data"
    invalid_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": "string_data"
    }
    response = client.post("/v1/predict", json=invalid_payload)
    # Утверждаем, что сервер заблокировал некорректный тип данных
    assert response.status_code == 422
    assert "unable to parse string as a number" in response.text

def test_predict_empty_payload_error(client):
    # Отправляем пустой словарь
    response = client.post("/v1/predict", json={})
    assert response.status_code == 422
    assert "Field required" in response.text
