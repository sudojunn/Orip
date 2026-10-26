from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. Загружаем полный датасет Iris (3 класса)
iris = load_iris()
X, y = iris.data, iris.target

print("Классы:", iris.target_names)        # ['setosa' 'versicolor' 'virginica']
print("Уникальные метки:", set(y))         # {0, 1, 2}

# 2. Обучаем модель на всех трёх классах
model = LogisticRegression(max_iter=500)
model.fit(X, y)

# 3. Конвертируем в ONNX
initial_type = [("float_input", FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={id(model): {"zipmap": True}},   # оставляем ZipMap, как в исходной модели
)

with open("iris_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Модель сохранена в iris_model.onnx")
