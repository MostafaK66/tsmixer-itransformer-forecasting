.PHONY: install test lint format typecheck quality

install:
	python -m pip install -e '.[all,dev]'

test:
	python -m pytest --cov=tsmixer_itransformer --cov-report=term-missing

lint:
	python -m ruff check src tests

format:
	python -m ruff format src tests

typecheck:
	python -m mypy

quality: lint typecheck test
	python -m compileall -q src tests
