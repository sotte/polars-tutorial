polars-tutorial.ipynb: polars-tutorial.py
	.venv/bin/jupytext --to ipynb --execute polars-tutorial.py

polars-tutorial.py: .venv
	.venv/bin/ruff format

.venv: requirements.txt
	python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.txt
