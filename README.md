# ML Project — Weather Rain Prediction

**Студент:** [Наумов Виталий Вячеславович / Student ID]

**Группа:** [БИВ234]

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт-локально)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

ML-проект для прогноза признака `RainTomorrow`: будет ли дождь на следующий день по данным о температуре, влажности, давлении, ветре, облачности и осадках

Датасет: **Weather Dataset Rattle Package** (`jsphyg/weather-dataset-rattle-package`) с Kaggle  
Данные скачиваются автоматически через `kagglehub`; CSV-файл не хранится в репозитории

**Целевая метрика:** [Accuracy / F1 / RMSE / ...]

## Что сделано по критериям

| Критерий | Где реализовано |
|---|---|
| Полная очистка: пропуски, дубли, выбросы, типы | `notebooks/01_full_weather_rain_prediction.ipynb`, `src/preprocessing.py` |
| Работа с фичами: исходные и новые признаки, feature engineering | `src/features.py`, notebook-раздел Feature Engineering |
| Визуализации зависимостей | notebook-раздел EDA + `src/visualize.py` |
| Корректный train/val/test split и защита от datalake/data leakage | chronological split по `Date`, удаление `RISK_MM`, fit preprocessing только на train |
| Выбор метрик и обоснование | notebook-раздел Metrics |
| Самостоятельный парсинг данных | `src/data_loading.py`, notebook: `kagglehub.dataset_download(...)` |
| Baseline без feature engineering | `Baseline LogisticRegression` |
| Минимум 4-5 моделей + ансамбли | Logistic Regression, KNN, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, Voting, Stacking |
| Эксперименты и перебор гиперпараметров | таблица экспериментов в notebook и `models/experiment_results.csv` после запуска |
| Уменьшение размерности | TruncatedSVD/PCA-раздел в notebook |
| Обоснование финальной модели | notebook-раздел Final model |
| Чистая структура проекта | см. структуру ниже |
| Линтеры | `ruff`, конфиг в `pyproject.toml` |
| Fixed seed | `RANDOM_STATE = 42` в `src/config.py` |
| requirements/pyproject с версиями | `requirements.txt`, `pyproject.toml` |
| Docker/docker-compose | `Dockerfile`, `docker-compose.yml` |
| Описание структуры и датасета | этот README + `data/README.md` |


## Структура репозитория

```text
weather-rain-repo/
├── .github/workflows/ci.yml
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── notebooks/
│   └── 01_full_weather_rain_prediction.ipynb
├── presentation/
│   └── README.md
├── report/
│   └── report.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── evaluate.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── utils.py
│   └── visualize.py
├── tests/
│   └── test_preprocessing.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Быстрый старт локально

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Проверка стиля и тестов
ruff check src tests
pytest -q

# Быстрый запуск экспериментов
python -m src.train --quick

# Полный запуск
python -m src.train
```

## Запуск ноутбука

```bash
jupyter notebook notebooks/01_full_weather_rain_prediction.ipynb
```

В ноутбуке данные скачиваются:

```python
import kagglehub

path = kagglehub.dataset_download("jsphyg/weather-dataset-rattle-package")
print("Path to dataset files:", path)
```

## Docker

Сборка:

```bash
docker compose build
```

Быстрый запуск обучения в контейнере:

```bash
docker compose run --rm train
```

Запуск Jupyter Notebook в контейнере:

```bash
docker compose up notebook
```

После запуска откройте:

```text
http://localhost:8888
```

Режим обучения модели в докере
```
docker run --rm -it \
  -v "$PWD":/app \
  -w /app \
  weather-rain \
  python -m src.train
```

Прогнать ноутбук автоматически и сохранить выводы в .ipynb
```
 docker run --rm -it \
  -v "$PWD":/app \
  -w /app \
  weather-rain \
  jupyter nbconvert \
    --to notebook \
    --execute notebooks/weather_rain_prediction.ipynb \
    --output weather_rain_prediction.ipynb \
    --output-dir notebooks
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты
Таблица с результатами находится в models/experiment_results.csv

После запуска обучения появятся файлы:

```text
models/best_model.joblib
models/experiment_results.csv
```

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
