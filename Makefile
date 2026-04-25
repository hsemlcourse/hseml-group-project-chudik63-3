.PHONY: install lint format test train notebook docker-build docker-train docker-notebook

install:
	pip install -r requirements.txt

lint:
	ruff check src tests

format:
	ruff format src tests

format-check:
	ruff format --check src tests

test:
	pytest -q

train:
	python -m src.train

train-quick:
	python -m src.train --quick

notebook:
	jupyter notebook notebooks/01_full_weather_rain_prediction.ipynb

docker-build:
	docker compose build

docker-train:
	docker compose run --rm train

docker-notebook:
	docker compose up notebook
