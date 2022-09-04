.PHONY: lint black test

lint:
	python -m flake8 src

black:
	python -m black --line-length 79 src/

test:
	pytest tests

coverage:
	pytest --cov=src tests/