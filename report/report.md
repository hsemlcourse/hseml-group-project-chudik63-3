# Отчет по проекту: прогноз дождя на следующий день

## 1. Цель

Построить модель машинного обучения, которая по погодным признакам определяет, ожидается ли дождь на следующий день (`RainTomorrow`)

## 2. Данные

Используется Kaggle dataset `jsphyg/weather-dataset-rattle-package`. Загрузка выполняется автоматически через `kagglehub`, что закрывает пункт самостоятельного парсинга/получения данных

Основной файл: `weatherAUS.csv`

## 3. Очистка данных

В проекте предусмотрены следующие шаги:

1. Удаление дубликатов
2. Приведение `Date` к типу `datetime`
3. Удаление строк без таргета `RainTomorrow`
4. Преобразование таргета `Yes/No` в `1/0`
5. Обработка пропусков:
   - числовые признаки: median imputation
   - категориальные признаки: most frequent imputation
6. Обработка выбросов:
   - clipping числовых признаков по train-квантилям 1% и 99%
7. Защита от leakage:
   - удаляется `RISK_MM`, так как это количество дождя на следующий день
   - все preprocessors обучаются только на train через `Pipeline`

## 4. Feature Engineering

Исходные признаки дополняются новыми:

- `Year`, `Month`, `DayOfYear`, `WeekOfYear`, `Season`
- `TempRange = MaxTemp - MinTemp`
- `TempMean`
- `HumidityChange = Humidity3pm - Humidity9am`
- `PressureChange = Pressure3pm - Pressure9am`
- `WindSpeedMean`
- `WindGustToMeanSpeed`
- `RainfallLog1p`
- `HadRainTodayByMm`
- `LowSunshine`

## 5. Split

Основной split — хронологический:

- train: первые 70% наблюдений по времени
- validation: следующие 15%
- test: последние 15%

## 6. Метрики

Главная метрика: **PR-AUC**

Обоснование: задача бинарной классификации с дисбалансом классов, положительный класс `RainTomorrow = Yes` встречается реже. PR-AUC лучше отражает качество на положительном классе, чем accuracy

Дополнительные метрики:

- ROC-AUC
- F1
- Recall
- Precision
- Accuracy
- Confusion Matrix

Threshold выбирается по validation-части по максимуму F1

## 7. Эксперименты

В проекте реализованы:

- baseline Logistic Regression без feature engineering
- Logistic Regression с feature engineering
- KNN
- RandomForest
- ExtraTrees
- GradientBoosting
- XGBoost
- LightGBM
- RandomForest с подбором гиперпараметров
- Soft Voting ensemble
- Stacking ensemble

Результаты сохраняются в `models/experiment_results.csv`

## 8. Уменьшение размерности

В ноутбуке есть эксперимент с `TruncatedSVD`, потому что после OneHotEncoding матрица признаков может стать широкой и разреженной. Используется:

- 2D-визуализация train объектов
- сравнение Logistic Regression без SVD и с SVD

## 9. Финальная модель

Финальная модель выбирается по максимальному validation PR-AUC. Test используется только для финальной оценки выбранного подхода
