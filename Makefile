default:

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install -r dev_requirements.txt
	pip install -r docs_requirements.txt

test: simba_ml tests
	coverage erase
	coverage run -m pytest tests -v
	coverage run -a -m pytest simba_ml --doctest-modules
	coverage report

lint: simba_ml
	pycodestyle --max-line-length=88 --ignore E203,W503 --select W504 simba_ml
	pylint simba_ml
	pydocstyle --convention=google simba_ml
	sourcery review simba_ml --check 
	mypy --pretty simba_ml/ --disable-error-code import --disable-error-code no-any-return --strict
	find simba_ml ! -iwholename "simba_ml\/\_version\.py" -name "*.py" | xargs darglint -v 2
	black simba_ml --check

tests-lint: simba_ml tests
	pycodestyle --max-line-length=88 --ignore E203,W503 --select W504 tests
	pylint tests --disable missing-module-docstring,missing-function-docstring,only-importing-modules-is-allowed
	sourcery review tests --check

documentation:
	cd docs && make html

doctest:
	cd docs && make doctest

check: lint tests-lint test

format:
	black simba_ml
	autopep8 simba_ml --ignore E203,W503 --select W504 --in-place --recursive

build: simba_ml
	python setup.py --command-packages=click_man.commands man_pages
	python setup.py sdist bdist_wheel