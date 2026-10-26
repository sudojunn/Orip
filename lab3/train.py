import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
# 1. Настройка подключения к нашему запущенному серверу MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# 2. Создание или выбор глобального эксперимента
experiment_name = "iris_classification_experiment"
mlflow.set_experiment(experiment_name)

def main():
    # Задаем гиперпараметры (их мы будем менять в Этапе 3.3)
    n_estimators = 50
    max_depth = 3
    random_state = 25
    # Открываем контекстный менеджер для старта одного уникального запуска (Run)
    with mlflow.start_run(run_name="RandomForest_Run"):
# --- Симуляция загрузки данных ---
# В реальной работе мы бы читали файл data/dataset.csv, версионированныйв DVC.
# Создадим синтетический датасет для простоты демонстрации
        np.random.seed(random_state)
        X = np.random.rand(200, 4)
        y = np.random.choice([0, 1], size=200)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.2, random_state=random_state)
        # --- Логирование входных параметров (Backend Store) ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset_version", "v2.0") # Пример связи с DVC
        # --- Обучение модели ---
        model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
        )
        model.fit(X_train, y_train)
        # --- Оценка и логирование метрик (Backend Store) ---
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        # Логируем финальные метрики качества
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        # Симулируем логирование метрики по эпохам (шагам), как в нейросетях

        for epoch in range(1, 6):
            # В реальной НС тут был бы валидационный лосс текущей эпохи
            fake_loss = 0.5 / epoch
            mlflow.log_metric("train_loss", fake_loss, step=epoch)
        print(f"Эксперимент завершен. Accuracy: {acc:.4f}, F1: {f1:.4f}")
        # --- Создание и логирование артефактов-картинок (Artifact Store) ---
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(5, 5))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), model.feature_importances_)
        plt.savefig("plots/feature_importance.png")
        plt.close()
        # Отправляем созданный график на сервер MLflow
        mlflow.log_artifact("plots/feature_importance.png")
        # --- Сериализация и сохранение самой модели в MLflow ---
        # Функция сохраняет модель в специальном внутреннем формате MLmodel
        mlflow.sklearn.log_model(model, artifact_path="random_forest_model")

if __name__ == "__main__":
    main()