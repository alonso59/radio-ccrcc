# Radiology WebUI

Radiology WebUI is a local-first radiology viewer for the `radio-ccrcc` dataset layout. It combines a FastAPI backend and a React + TypeScript + Vite frontend for dataset browsing, slice review, overlay visualization, 3D mesh rendering, and reviewer decision workflows.

See the full requirements in [docs/SRS.md](docs/SRS.md).

## Environment

- Backend runtime: Anaconda environment `ccrcc`
- Frontend runtime: vendored `udocker` with a `node:20-slim` image
- Port forwarding: handled by VS Code Remote SSH
- Portable runtime for other machines: native Docker + `docker compose`
- Frontend auth token storage defaults to in-memory (`frontend/.env.example`)

## Local udocker entrypoints

This repo ships two local entrypoints:

- `udocker-1.3.17/udocker/udocker` (upstream launcher)
- `udocker.py` (project wrapper that sets `UDOCKER_DIR`, `PYTHONPATH`, and `PROOT_NO_SECCOMP`)

Examples:

```bash
python udocker.py version
python udocker.py images -l
```

## Quickstart

```bash
make setup-node
make install-backend
make dev-backend
make dev-frontend
```

To produce the frontend static bundle with udocker:

```bash
make build-frontend
```

## Manual Verification

- Backend health: `http://localhost:8000/api/health`
- Frontend dev server: `http://localhost:5173`

## Review Workflow Safety

When `ALLOW_DATA_MUTATIONS=true`, `Apply Changes` in the viewer mutates dataset files:

- Reclassify (NIfTI): updates `manifest.csv` (`phase`, `protocol_source=manual`)
- Reclassify/Delete (VOI): moves files to phase folders or recycle paths
- Delete (NIfTI): moves image/seg files to recycle paths

Audit files are appended in each dataset root:

- `decisions.json`
- `reclassification_log.json`
- `deletion_log.json`

Recycle paths used by the app:

- `<dataset>/deleted/...`
- `<dataset>/voi/deleted/...`

Back up datasets before enabling mutations.

## Run On Other Machines (Docker Compose)

Yes, this repo can run on other machines without `udocker`.

1. Copy [.env.example](.env.example) to `.env`.
2. Set `DATASET_DIR` in `.env` to the host path that contains your dataset root.
3. Keep `ALLOW_DATA_MUTATIONS=false` for read-only mode, or set it to `true` to enable reviewer apply operations.
4. Start the app:

```bash
docker compose up -d --build
```

5. Open:

- `http://localhost:8000/`
- `http://localhost:8000/api/health`

Useful commands:

```bash
docker compose logs -f
docker compose down
```

Equivalent Make targets:

```bash
make compose-build
make compose-up
make compose-logs
make compose-down
```

## OCI Build (M12)

The project now includes a multi-stage [Dockerfile](Dockerfile):

- Stage 1 builds the Vite frontend.
- Stage 2 installs backend dependencies and serves `/app/static` via FastAPI.
- Container default dataset root is `DATA_ROOT=/data`.

Build the image with Podman:

```bash
podman build -t radiology-ui:1.0 .
```

Build the same image with native Docker:

```bash
docker build -t radiology-ui:1.0 .
```

Run it against the mounted dataset:

```bash
podman run --rm -p 8000:8000 \
  -e ALLOW_DATA_MUTATIONS=true \
  -v /home/alonso/Documents/radio-ccrcc/data/dataset:/data:rw \
  radiology-ui:1.0
```

Docker equivalent:

```bash
docker run --rm -p 8000:8000 \
  -e ALLOW_DATA_MUTATIONS=true \
  -v /home/alonso/Documents/radio-ccrcc/data/dataset:/data:rw \
  radiology-ui:1.0
```

Then open:

- `http://localhost:8000/`
- `http://localhost:8000/api/health`

## udocker Deployment

Export the built OCI image and run with udocker:

```bash
podman save -o radiology-ui_1.0.tar radiology-ui:1.0
# or: docker save -o radiology-ui_1.0.tar radiology-ui:1.0
python udocker.py load -i radiology-ui_1.0.tar
python udocker.py create --name=radio-ui radiology-ui:1.0
python udocker.py run -p 8000:8000 \
  -e ALLOW_DATA_MUTATIONS=true \
  -v /home/alonso/Documents/radio-ccrcc/data/dataset:/data:rw \
  radio-ui
```

The repo includes a local `udocker` bundle and stores its runtime state under `.udocker/`, so no global `udocker` command on `PATH` is required.
