# Radiology WebUI

Radiology WebUI is a local-first radiology viewer for the `radio-ccrcc` dataset layout, with a FastAPI backend and a React + TypeScript + Vite frontend intended for interactive dataset browsing and image review on a remote Linux server.

See the full requirements in [docs/SRS.md](docs/SRS.md).

## Environment

- Backend runtime: Anaconda environment `ccrcc`
- Frontend runtime: vendored `udocker` with a `node:20-slim` image
- Port forwarding: handled by VS Code Remote SSH

## Quickstart

```bash
make setup-node
make install-backend
make dev-backend
make dev-frontend
```

## Manual Verification

- Backend health: `http://localhost:8000/api/health`
- Frontend dev server: `http://localhost:5173`

## Notes

Docker and OCI packaging are deferred to Milestone 12.

The repo includes a local `udocker` bundle and stores its runtime state under `.udocker/`, so no global `udocker` command on `PATH` is required.
