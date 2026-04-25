# Data

Данные **не хранятся в репозитории**. Они скачиваются автоматически из Kaggle через `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("jsphyg/weather-dataset-rattle-package")
print("Path to dataset files:", path)
```

Ожидаемый основной файл датасета: `weatherAUS.csv`

В Docker и локальном запуске данные будут скачаны в кэш KaggleHub
