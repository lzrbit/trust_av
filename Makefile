PYTHON ?= python
WORKSPACE ?= work
FONT ?= Arial

.PHONY: map check prepare figures tables panels examples smoke
map:
	$(PYTHON) manuscript.py list
check:
	$(PYTHON) manuscript.py --workspace "$(WORKSPACE)" check
prepare:
	$(PYTHON) manuscript.py --workspace "$(WORKSPACE)" prepare
figures:
	$(PYTHON) manuscript.py --workspace "$(WORKSPACE)" --font "$(FONT)" figures
panels:
	$(PYTHON) manuscript.py --workspace "$(WORKSPACE)" panels
tables:
	$(PYTHON) manuscript.py --workspace "$(WORKSPACE)" tables
examples:
	$(PYTHON) run.py --workspace "$(WORKSPACE)" aggregate-figures
smoke:
	$(PYTHON) tests/smoke.py
