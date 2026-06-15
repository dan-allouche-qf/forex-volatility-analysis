.PHONY: setup lint typecheck test cov run clean kernel snapshot report-guard app docs help

VENV ?= venv
PY   := $(VENV)/bin/python

setup:  ## create venv and install the package + dev/app/docs tools
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e ".[dev,app,docs]"

lint:  ## static checks (ruff)
	$(VENV)/bin/ruff check src tests

typecheck:  ## mypy on the package
	$(PY) -m mypy

test:  ## run the test suite
	$(PY) -m pytest

cov:  ## test suite with coverage
	$(PY) -m pytest --cov=fxvol --cov-report=term-missing

report-guard:  ## fail if any committed number drifts from the computed value
	$(PY) scripts/check_numbers.py

app:  ## launch the interactive Streamlit dashboard
	$(VENV)/bin/streamlit run app/streamlit_app.py

docs:  ## build the documentation site
	$(VENV)/bin/mkdocs build --strict

kernel:  ## register a Jupyter kernel for this venv (named "fxvol")
	$(PY) -m ipykernel install --user --name fxvol --display-name "fxvol"

run: kernel  ## execute the notebook end-to-end (refreshes figures/, proves reproducibility)
	$(VENV)/bin/jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=fxvol \
		--output /tmp/fxvol_executed.ipynb notebooks/fx_volatility_analysis.ipynb

snapshot:  ## re-pull the FX data snapshot from Yahoo Finance (overwrites data/)
	$(PY) -c "from fxvol import data; data.load_ohlc(refresh=True)"

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache src/*.egg-info /tmp/fxvol_executed.ipynb
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'
