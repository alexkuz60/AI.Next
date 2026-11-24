.PHONY: install lint fmt test run

install:
	.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

lint:
	.venv/bin/python -m ruff check .

fmt:
	.venv/bin/python -m ruff format .

test:
	.venv/bin/python -m pytest

run:
	.venv/bin/python app_next.py
