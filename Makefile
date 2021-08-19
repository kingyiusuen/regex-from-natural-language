install:
	python -m pip install -e . --no-cache-dir

install-dev:
	python -m pip install -e ".[dev]" --no-cache-dir
	pre-commit install
	pre-commit autoupdate

venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	make install-dev

lint:
	isort .
	black .
	flake8 .
	mypy .

clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . -type f -name ".coverage*" -ls -delete
	rm -rf htmlcov
	rm -rf .mypy_cache
