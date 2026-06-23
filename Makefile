.PHONY: check format lint test

format:
	python -m ruff check --fix .
	python -m ruff format .

lint:
	python -m ruff check .
	python -m ruff format --check .
	python -m mypy

test:
	python -m pytest

check: lint test

