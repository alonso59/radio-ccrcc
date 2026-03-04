SHELL := /bin/bash

UDOCKER_DIR := /home/alonso/Documents/radio-ccrcc/radioccrcc-webui/.udocker
UDOCKER_PYTHON := /home/alonso/anaconda3/envs/ccrcc/bin/python
UDOCKER_PYTHONPATH := /home/alonso/Documents/radio-ccrcc/radioccrcc-webui/udocker-1.3.17
UDOCKER_MAIN := /home/alonso/Documents/radio-ccrcc/radioccrcc-webui/udocker-1.3.17/udocker/maincmd.py
UDOCKER = PROOT_NO_SECCOMP=1 UDOCKER_DIR=$(UDOCKER_DIR) PYTHONPATH=$(UDOCKER_PYTHONPATH) $(UDOCKER_PYTHON) $(UDOCKER_MAIN)

.PHONY: setup-node install-backend dev-backend dev-frontend

setup-node:
	$(UDOCKER) pull node:20-slim
	$(UDOCKER) create --name=radio-node node:20-slim

install-backend:
	source /home/alonso/anaconda3/etc/profile.d/conda.sh && \
	conda activate ccrcc && \
	cd backend && \
	pip install -r requirements.txt

dev-backend:
	source /home/alonso/anaconda3/etc/profile.d/conda.sh && \
	conda activate ccrcc && \
	cd backend && \
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	$(UDOCKER) run --hostenv \
		-v /home/alonso/Documents/radio-ccrcc/radioccrcc-webui/frontend:/app \
		-p 5173:5173 \
		radio-node \
		bash -c "cd /app && npm install && npm run dev -- --host 0.0.0.0 --port 5173"
