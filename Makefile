.PHONY: lint black test

lint:
	python -m flake8 nlp_sa

black:
	python -m black --line-length 70 nlp_sa/utils

test:
	pytest tests

coverage:
	pytest --cov=src tests/