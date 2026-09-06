# MLflow: единый sqlite-бэкенд для train и ui (.env переопределяет дефолт)
-include .env
MLFLOW_TRACKING_URI ?= sqlite:///$(CURDIR)/mlflow.db
export MLFLOW_TRACKING_URI

reference:
	uv run python -m pubg_ds.features.reference

collect:
	uv run python -m pubg_ds.collect.matches

live:
	uv run python -m pubg_ds.features.live

monitor:
	uv run python -m pubg_ds.monitor.drift
	uv run python -m pubg_ds.monitor.performance

tune:
	uv run python -m pubg_ds.models.tune

train:
	uv run python -m pubg_ds.models.train

retrain:
	uv run python -m pubg_ds.models.retrain

ui:
	uv run mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI)

test:
	uv run pytest -q

.PHONY: reference collect live monitor tune train retrain ui test
