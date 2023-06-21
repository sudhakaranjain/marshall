VENV_DIR = ./venv
PIP = $(VENV_DIR)/bin/pip
PYTHON = $(VENV_DIR)/bin/python

clean-venv: ## Remove venv
	if [ -d "./venv" ]; then \
        rm -rf venv; \
    fi

create-venv: clean-venv
	python3.8 -m venv $(VENV_DIR) --system-site-packages
	$(PIP) install --upgrade pip setuptools wheel

install: create-venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
