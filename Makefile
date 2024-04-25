.PHONY: help requirements install format lint test coverage docs

ifndef VIRTUAL_ENV
ifdef CONDA_PREFIX
$(warning "For better compatibility, consider using a plain Python venv instead of Conda")
VIRTUAL_ENV := $(CONDA_PREFIX)
else
$(error "This Makefile needs to be run inside a virtual environment")
endif
endif

SHELL := /bin/bash
PLATFORM := $(shell \
  if [[ -n $$(command -v nvidia-smi && nvidia-smi --list-gpus) ]]; then echo cu118; \
  else echo cpu; \
  fi)

help:
	@echo "Available commands:"
	@echo "install            install dev requirements."
	@echo "format             format code."
	@echo "lint               run linters."
	@echo "test               run unit tests."
	@echo "coverage           build coverage report."
	@echo "docs               build documentation."

$(VIRTUAL_ENV)/timestamp: requirements/*.in
	@touch $(VIRTUAL_ENV)/timestamp

install: $(VIRTUAL_ENV)/timestamp
	python -m pip install -q --upgrade pip wheel 
	pip install -q --extra-index-url https://download.pytorch.org/whl/$(PLATFORM) -e '.[dev]'

format: install
	tox run -e format

lint: install
	tox run -e lint

test: install
	tox run -f test

coverage: install
	tox run -e coverage

docs: install
	tox run -e docs