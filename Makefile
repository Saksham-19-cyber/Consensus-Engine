.PHONY: test build benchmarks

test:
	pytest

build:
	cd web && npm run build

benchmarks:
	python scripts/run_all_benchmarks.py
