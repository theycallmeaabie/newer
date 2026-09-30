# Common tasks. Everything runs offline once `make setup` has fetched dependencies.
PY      ?= python3
VENV    := .venv
BIN     := $(VENV)/bin
API_PORT ?= 8000

# A half-finished recipe must not leave its target looking up to date (e.g. pip failing mid-way).
.DELETE_ON_ERROR:
STAMP    := $(VENV)/.deps-installed

.PHONY: help setup setup-py setup-web lock test lint format typecheck api web web-build demo clean

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: setup-py setup-web ## create the venv, install Python and web dependencies

setup-py: $(STAMP)

$(STAMP): pyproject.toml requirements.lock
	test -x $(BIN)/python || $(PY) -m venv $(VENV)
	$(BIN)/python -c 'import sys; assert sys.version_info >= (3, 12), sys.version'
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]" -c requirements.lock
	touch $@

lock: setup-py ## regenerate requirements.lock from the current venv (after changing pyproject deps)
	$(BIN)/pip freeze --exclude-editable > requirements.lock

# An installed binary is the sentinel: it only exists when the tree is intact.
WEB_STAMP := web/node_modules/.bin/vite

setup-web: $(WEB_STAMP)

$(WEB_STAMP): web/package.json web/package-lock.json
	cd web && npm ci --no-audit --no-fund
	touch $@

test: setup-py ## run the test suite
	$(BIN)/pytest

lint: setup-py ## ruff lint + format check (Python) and tsc (web)
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: setup-py ## auto-format and fix lint
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck: setup-web ## TypeScript type check of the dashboard
	cd web && npm run typecheck

api: setup-py ## run the FastAPI server with reload
	$(BIN)/uvicorn api.main:app --host 127.0.0.1 --port $(API_PORT) --reload

web: setup-web ## run the Vite dev server (proxies /api to the FastAPI server)
	cd web && npm run dev

web-build: setup-web ## build the dashboard into web/dist (served by the API)
	cd web && npm run build

demo: ## one command: generate data -> train -> serve -> open dashboard
	bash scripts/demo.sh

clean: ## remove build artefacts (keeps the venv and node_modules)
	rm -rf web/dist .pytest_cache .ruff_cache build *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
