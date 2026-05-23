.PHONY: help install lint format type test test-fast smoke compile clean docker-up docker-down

PYTHON := python
PIP    := $(PYTHON) -m pip

help:
	@echo "Targets:"
	@echo "  install      install pipeline + dev dependencies"
	@echo "  lint         ruff check"
	@echo "  format       ruff format"
	@echo "  type         mypy (non-blocking)"
	@echo "  test         full pytest"
	@echo "  test-fast    pytest skipping airflow + slow"
	@echo "  smoke        compileall + import smoke + lightweight tests"
	@echo "  compile      python -m compileall src scripts dags tests"
	@echo "  docker-up    docker compose up"
	@echo "  docker-down  docker compose down"
	@echo "  clean        remove __pycache__ and .pytest_cache"

install:
	$(PIP) install -r requirements-pipeline.txt
	$(PIP) install -r dashboards/requirements-dashboards.txt
	$(PIP) install pytest pytest-cov ruff mypy pre-commit

lint:
	ruff check src scripts dags tests

format:
	ruff format src scripts dags tests

type:
	mypy src --ignore-missing-imports --no-strict-optional || true

test:
	pytest -v

test-fast:
	pytest -v -m "not slow and not requires_airflow"

compile:
	$(PYTHON) -m compileall -q src scripts dags tests

smoke: compile
	$(PYTHON) -c "from src import paths, config, validation, idempotency, feature_engineering, inference, network_kpis, sampling; print('imports OK')"
	pytest -q tests/test_paths.py tests/test_config.py tests/test_validation.py tests/test_feature_engineering.py tests/test_inference.py tests/test_idempotency.py tests/test_network_kpis.py

docker-up:
	docker compose build
	docker compose up airflow-init
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
