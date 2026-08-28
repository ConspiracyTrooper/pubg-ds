reference:
	uv run python -m pubg_ds.features.reference

test:
	uv run pytest -q

.PHONY: reference test