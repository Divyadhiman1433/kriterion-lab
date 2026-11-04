.PHONY: install test lint experiments app docker-build

install:
	python -m pip install -e ".[dev]"

test:
	pytest --cov=kriterion --cov-report=term-missing

lint:
	ruff check .

experiments:
	python scripts/run_experiments.py

app:
	streamlit run app.py

docker-build:
	docker build -t kriterion-ml-lab .

