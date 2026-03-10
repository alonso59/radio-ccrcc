SHELL := /bin/bash

PROJECT_ROOT := /home/alonso/Documents/radio-ccrcc/radioccrcc-webui
UDOCKER := /home/alonso/anaconda3/envs/ccrcc/bin/python $(PROJECT_ROOT)/udocker.py

.PHONY: setup-node install-backend dev-backend dev-frontend build-frontend compose-build compose-up compose-down compose-logs

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

build-frontend:
	$(UDOCKER) run --hostenv \
		-v /home/alonso/Documents/radio-ccrcc/radioccrcc-webui/frontend:/app \
		radio-node \
		bash -c "cd /app && npm install && npm run build"

compose-build:
	docker compose build

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f
