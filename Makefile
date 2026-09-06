# MLflow: единый sqlite-бэкенд для train и ui (.env переопределяет дефолт)
-include .env
MLFLOW_TRACKING_URI ?= sqlite:///$(CURDIR)/mlflow.db
export MLFLOW_TRACKING_URI

reference:
	uv run python -m pubg_ds.features.reference

collect:
	uv run python -m pubg_ds.collect.matches

tune:
	uv run python -m pubg_ds.models.tune

train:
	uv run python -m pubg_ds.models.train

ui:
	uv run mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI)

test:
	uv run pytest -q

.PHONY: reference collect tune train ui test
