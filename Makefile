clean:
	find src -type f -name '*.so' -delete
	find src -type f -name '*.c' -delete

build:
	rm -rf dist
	pip install build
	python -m build