# AGENTS.md — Development Plan for Radiology WebUI

> **Purpose**: This file defines a milestone-based plan that an AI coding agent
> can follow step-by-step. Each milestone is self-contained, produces a
> verifiable deliverable, and includes explicit completion criteria so the agent
> (or a human reviewer) knows when to move on.
>
> **Rule**: Never start a milestone until the previous one's completion criteria
> are fully met. Mark each task with `[x]` as it is finished.

---

## ⚠ Environment Constraints (READ BEFORE ANYTHING ELSE)

This project runs on a **remote Linux server with no sudo access**.

| Layer | Runtime | Notes |
|---|---|---|
| **Python backend** | **Anaconda env `ccrcc`** (native) | `conda activate ccrcc` then `pip install` or `conda install` |
| **Frontend dev** | **`udocker` + `node:20-slim` image** | Node.js is NOT available natively on the server |
| **Data access** | Native filesystem | Direct path access, no container needed |
| **Port forwarding** | **VS Code Remote SSH** | Already configured — do NOT suggest ssh tunnel commands |
| **Container deploy** | **udocker** | Final deployment only (M12) — NOT the dev environment |

### Dev loop (always use this exact pattern)

```bash
# Terminal 1 — Backend (native Anaconda)
conda activate ccrcc
cd /home/alonso/Documents/radio-ccrcc/radioccrcc-webui/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend (udocker node container)
udocker run --hostenv \
  -v /home/alonso/Documents/radio-ccrcc/radioccrcc-webui/frontend:/app \
  -p 5173:5173 \
  radio-node \
  bash -c "cd /app && npm install && npm run dev -- --host 0.0.0.0 --port 5173"
```

### One-time udocker node setup (M0 only, run once on the server)
```bash
udocker pull node:20-slim
udocker create --name=radio-node node:20-slim
```

### Hard constraints for the agent
- ❌ Never use `sudo`, `apt`, `brew`, or system-level installs
- ❌ Never suggest native `npm` or `node` commands outside udocker
- ❌ Never modify files outside `radioccrcc-webui/`
- ❌ Never suggest SSH port-forward commands (VS Code handles this)
- ✅ Always use `conda activate ccrcc` before backend commands
- ✅ Always use udocker for any Node.js operation
- ✅ Docker/Podman and Dockerfile are deferred to **M12 only**

---

## Project Layout

```
radioccrcc-webui/
├── AGENTS.md              ← you are here
├── docs/
│   └── SRS.md             ← full requirements specification
├── backend/               ← Python FastAPI — created in M0
├── frontend/              ← React + TypeScript + Vite — created in M0
├── Makefile               ← dev commands — created in M0
├── .gitignore             ← created in M0
├── Dockerfile             ← created in M12 only
└── README.md              ← created in M0
```

The parent project `radio-ccrcc/` provides the data (`data/dataset/`) and
reference code (`notebooks/`, `src/converter/`, `src/preprocessor/`).
This WebUI project is **isolated**: it has its own dependencies and can be
built independently.

---

## Reference Files (parent repo — read-only, never modify)

These files contain patterns, conventions, and utilities the agent should
read when implementing specific milestones. All paths are relative to
`/home/alonso/Documents/radio-ccrcc/`.

| Absolute path                                                                 | Relevant to | Contains                                           |
|-------------------------------------------------------------------------------|-------------|----------------------------------------------------|
| `notebooks/utils/visualizer_utils.py`                                         | M2, M3      | HU normalize, overlay_multi_layer_mask, crosshair colors, render_orthogonal_views |
| `notebooks/nifti_visualizer.ipynb`                                            | M2          | NIfTI load + RAS reorient + seg overlay pattern (Cell 3) |
| `notebooks/voi_visualizer.ipynb`                                              | M2          | VOI .npy load + mask overlay pattern (Cell 3)      |
| `src/preprocessor/discover.py`                                                | M1          | Manifest parsing, patient ID extraction regex      |
| `src/preprocessor/pipeline.py`                                                | M1          | VOI path convention: `{type}/{group}/{pid}/{phase}/{name}.npy` |
| `src/converter/pipeline.py`                                                   | M1          | NIfTI naming: `{NN}_case_{YYYYY}_0000.nii.gz`, manifest.csv schema |
| `data/dataset/Dataset820/`                                                    | M1, M2      | Real dataset: `nifti/`, `seg/`, `voi/`, `manifest.csv` |

---

## Milestones

### Milestone 0 — Project Bootstrap

**Goal**: Both servers start without errors. Dev loop works end-to-end.

**Environment reminder**: Backend = Anaconda `ccrcc`. Frontend = udocker `radio-node`. No Docker/Dockerfile yet.

**Tasks**:

- [x] **M0.1** Pull node image and create udocker container (one-time):
  ```bash
  udocker pull node:20-slim
  udocker create --name=radio-node node:20-slim
  ```
- [x] **M0.2** Create `backend/` skeleton:
  - `backend/requirements.txt` — packages compatible with Anaconda `ccrcc`:
    `fastapi`, `uvicorn[standard]`, `nibabel`, `numpy`, `scikit-image`, `Pillow`, `trimesh`, `pydantic`, `python-multipart`
  - `backend/app/__init__.py`
  - `backend/app/main.py` — minimal FastAPI app, `GET /api/health` → `{"status": "ok"}`
  - `backend/app/config.py` — reads env vars: `DATA_ROOT` (default `../../data/dataset`), `RADIOLOGY_UI_TOKEN`, `LOG_LEVEL`, `PORT`
- [x] **M0.3** Install backend dependencies into Anaconda env:
  ```bash
  conda activate ccrcc
  pip install -r backend/requirements.txt
  ```
- [x] **M0.4** Create `frontend/` skeleton using udocker node container:
  ```bash
  udocker run --hostenv \
    -v /home/alonso/Documents/radio-ccrcc/radioccrcc-webui:/workspace \
    radio-node \
    bash -c "cd /workspace && npm create vite@latest frontend -- --template react-ts --yes"
  ```
- [x] **M0.5** Add Vite proxy config in `frontend/vite.config.ts`:
  ```ts
  server: { proxy: { '/api': 'http://localhost:8000' } }
  ```
  Also set `server.host: '0.0.0.0'` and `server.port: 5173`.
- [x] **M0.6** Create `Makefile` with targets:
  - `setup-node` — runs the udocker pull + create commands
  - `dev-backend` — `conda activate ccrcc && cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
  - `dev-frontend` — udocker run command with volume mount and port 5173
  - `install-backend` — `conda activate ccrcc && pip install -r backend/requirements.txt`
- [x] **M0.7** Create `.gitignore` covering: `node_modules/`, `__pycache__/`, `*.pyc`, `.env`, `dist/`, `*.tar`, udocker artifacts
- [x] **M0.8** Create `README.md` with project description, dev loop instructions, and link to `docs/SRS.md`
- [x] **M0.9** Verify backend starts:
  ```bash
  conda activate ccrcc && cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  # curl http://localhost:8000/api/health → {"status":"ok"}
  ```
- [x] **M0.10** Verify frontend starts inside udocker node container and browser loads Vite default page via VS Code port forward
- [x] **M0.11** Update Progress Tracker below

**Completion criteria**: `GET /api/health` returns `{"status":"ok"}` from the browser via VS Code port forward. Vite dev server loads at `http://localhost:5173`. No native Node.js commands were used.

---

### Milestone 1 — Backend: Data Discovery Service

**Goal**: Implement the folder-scanning logic that discovers datasets, patients,
series, and masks from the mounted data path.

**Tasks**:

- [x] **M1.1** Create `backend/app/services/discovery.py`:
  - `list_datasets(data_root) → list[DatasetSummary]`
    Scan `data_root/` for subdirectories matching `Dataset*`.
    For each, detect presence of `nifti/`, `seg/`, `voi/`, `manifest.csv`.
  - `discover_patients(dataset_path) → list[PatientSummary]`
    Parse `manifest.csv` if exists; else fall back to filename regex
    `(case_\d{5})`. Group by patient_id. Count series, seg, voi per patient.
  - `discover_series(dataset_path, patient_id) → list[SeriesInfo]`
    List NIfTI volumes for this patient. For each, check if matching seg exists
    (strip `_0000`). Check if VOI crops exist under `voi/images/`.
    Return list of series with type (nifti/voi), phase, has_seg flag, paths.
- [x] **M1.2** Create Pydantic models in `backend/app/models/`:
  - `dataset.py`: `DatasetSummary`, `PatientSummary`, `SeriesInfo`, `VolumeInfo`
  - Use `from __future__ import annotations` for forward refs
- [x] **M1.3** Create API routes in `backend/app/api/datasets.py`:
  - `GET /api/datasets` → calls `list_datasets`
  - `GET /api/datasets/{dsid}/patients` → calls `discover_patients`
  - `GET /api/datasets/{dsid}/patients/{pid}/series` → calls `discover_series`
- [x] **M1.4** Write tests (or manual curl verification):
  - Point `DATA_ROOT` at `../../data/dataset` (parent repo)
  - Verify Dataset820 is discovered with correct patient count
  - Verify `case_00001` returns series with NIfTI path + seg flag

**Completion criteria**: All three endpoints return correct JSON for Dataset820.
Discovery handles datasets with only `nifti/`, with `nifti/ + seg/`, and with
`voi/` present. Manifest-less datasets fall back to filename regex.

---

### Milestone 2 — Backend: Volume Loading & Slice API

**Goal**: Load NIfTI/NumPy volumes into memory and serve 2D slices as PNG.

**Tasks**:

- [ ] **M2.1** Create `backend/app/services/nifti_loader.py`:
  - `load_nifti(path) → (np.ndarray, spacing_tuple)`
    Uses `nibabel` + `as_closest_canonical` for RAS reorientation.
    Returns float32 array shape (X, Y, Z) + spacing (sx, sy, sz).
  - Reference: `nifti_visualizer.ipynb` Cell 3 (`CaseReviewState.load_current`)
- [ ] **M2.2** Create `backend/app/services/numpy_loader.py`:
  - `load_numpy(path) → np.ndarray`
    `np.load(path)` returning shape (X, Y, Z).
  - Reference: `voi_visualizer.ipynb` Cell 3
- [ ] **M2.3** Create `backend/app/services/mask_loader.py`:
  - `load_mask(path, is_nifti: bool) → np.ndarray`
    Load seg (.nii.gz) or voi mask (.npy). Handle 4D by taking `[..., 0]`.
    Return uint8 array.
  - Reference: `nifti_visualizer.ipynb` Cell 3 (seg loading block)
- [ ] **M2.4** Create `backend/app/services/slice_renderer.py`:
  - `render_slice(volume, mask, axis, index, ww, wl, layers, layer_config) → bytes`
    Extract 2D slice (axial/coronal/sagittal), apply W/L normalization,
    optionally overlay mask contours + fill, encode as PNG.
  - HU normalization: reference `visualizer_utils.py` → `hu_to_display()`
  - Overlay: reference `visualizer_utils.py` → `overlay_multi_layer_mask()`
  - Use `Pillow` for PNG encoding.
- [ ] **M2.5** Create `backend/app/services/volume_cache.py`:
  - In-memory cache holding current volume + mask (max 2 volumes).
  - `load_series(dataset_id, patient_id, series_id) → VolumeInfo`
  - `get_current() → (volume, mask, spacing)`
  - Evicts on new series load.
- [ ] **M2.6** Create API routes in `backend/app/api/slices.py`:
  - `POST /api/datasets/{dsid}/patients/{pid}/series/{sid}/load`
    → Load volume into cache, return `VolumeInfo {shape, spacing, has_mask, labels}`.
  - `GET /api/slice/{axis}/{index}?ww=400&wl=50&layers=1,2&opacity_1=0.15&opacity_2=0.20`
    → Return `image/png`.
- [ ] **M2.7** Manual verification:
  - Load a Dataset820 NIfTI series via POST
  - Fetch axial slice 100 as PNG, open in viewer
  - Fetch coronal slice with tumor overlay visible

**Completion criteria**: Slice API returns valid PNGs for all three axes.
Overlay contours match the visual output of the Jupyter notebooks.
VOI .npy series also render correctly through the same endpoints.

---

### Milestone 3 — Backend: 3D Mesh Generation

**Goal**: Generate surface meshes from segmentation masks and serve as GLB.

**Tasks**:

- [ ] **M3.1** Create `backend/app/services/mesh_generator.py`:
  - `generate_mesh(mask, label, spacing, smooth=True) → bytes`
    Use `skimage.measure.marching_cubes` to extract isosurface for given label.
    Optionally smooth with Laplacian.
    Export as GLB binary using `trimesh`.
  - Handle empty mask (no voxels for label) → return empty/null indicator.
- [ ] **M3.2** Create API route in `backend/app/api/mesh.py`:
  - `GET /api/mesh/{label}?smooth=true` → return `model/gltf-binary` or 404
- [ ] **M3.3** Verification:
  - Load series with seg, request mesh for label 2 (tumor)
  - Open returned GLB in an online viewer (e.g., gltf-viewer.donmccurdy.com)
  - Verify mesh geometry matches tumor shape

**Completion criteria**: GLB meshes are valid and loadable. Labels 1, 2, 3
each produce correct surfaces. Missing labels return 404.

---

### Milestone 4 — Backend: Settings & Auth

**Goal**: Implement simple token auth and persistent user settings.

**Tasks**:

- [ ] **M4.1** Create `backend/app/middleware/auth.py`:
  - If `RADIOLOGY_UI_TOKEN` is set (non-empty), require `Authorization: Bearer <token>`
    header on all `/api/*` routes except `/api/health`.
  - If token is empty, skip auth (local dev mode).
- [ ] **M4.2** Create `backend/app/services/settings_store.py`:
  - Read/write `viewer_settings.json` in a writable directory (not on mounted data).
  - Default location: `~/.radiology-webui/settings.json` or `/tmp/radiology-webui/settings.json` inside container.
  - Schema: `{dataset_id: {last_patient, last_series, ww, wl, layers_visible, layers_opacity}}`
- [ ] **M4.3** Create API routes in `backend/app/api/settings.py`:
  - `GET /api/settings` → current settings
  - `PUT /api/settings` → save settings
- [ ] **M4.4** Wire auth middleware into `main.py`.
- [ ] **M4.5** Verification:
  - Set `RADIOLOGY_UI_TOKEN=test123`, verify 401 without token, 200 with token
  - PUT settings, restart server, GET settings → persisted

**Completion criteria**: Auth blocks unauthenticated requests when token is set.
Settings persist across server restarts.

---

### Milestone 5 — Frontend: Project Setup & Routing

**Goal**: Set up the React/TypeScript frontend with pages, routing, dark theme,
and API service layer.

**Tasks**:

- [ ] **M5.1** Install dependencies:
  - `react-router-dom` (routing)
  - `@mui/material` + `@emotion/react` + `@emotion/styled` (UI components, dark theme)
  - `axios` (HTTP client)
- [ ] **M5.2** Create dark theme in `frontend/src/styles/theme.ts`:
  - Background `#1e1e1e`, panels `#121212`, borders `#333`, text `#ddd`
- [ ] **M5.3** Create typed API client in `frontend/src/services/api.ts`:
  - Functions for all backend endpoints
  - TypeScript interfaces matching Pydantic models
  - Auth token injection from localStorage
- [ ] **M5.4** Create route structure in `App.tsx`:
  - `/` → `DatasetSelectorPage`
  - `/datasets/:dsid/patients` → `PatientListPage`
  - `/datasets/:dsid/patients/:pid/viewer` → `ViewerPage`
- [ ] **M5.5** Create stub pages (empty dark containers with titles):
  - `DatasetSelectorPage.tsx`
  - `PatientListPage.tsx`
  - `ViewerPage.tsx`
- [ ] **M5.6** Verification:
  - Navigate between all three pages in browser
  - Dark theme applied consistently
  - API calls reach backend through Vite proxy

**Completion criteria**: Three pages render with dark theme. Navigation works.
API client successfully calls `/api/health`.

---

### Milestone 6 — Frontend: Dataset & Patient Pages

**Goal**: Build the dataset selector and patient list pages with real data.

**Tasks**:

- [ ] **M6.1** Implement `DatasetSelectorPage`:
  - Fetch `GET /api/datasets`
  - Display cards: dataset name, patient count, icons for nifti/seg/voi presence
  - Click card → navigate to patient list
- [ ] **M6.2** Implement `PatientListPage`:
  - Fetch `GET /api/datasets/{dsid}/patients`
  - Display sortable table: patient_id, group, phase(s), #series, #seg, #voi
  - Search bar filtering by patient_id
  - Group and phase filter dropdowns
  - Click patient row → navigate to viewer
- [ ] **M6.3** Create `LoginDialog` component:
  - If API returns 401, show token input dialog
  - Store token in localStorage, retry request
- [ ] **M6.4** Verification:
  - Dataset820 appears as card with correct metadata
  - Patient table populates, search/filter works
  - Clicking a patient navigates to `/datasets/Dataset820/patients/case_00001/viewer`

**Completion criteria**: Real data from Dataset820 renders in both pages.
Search and filter work. Auth dialog appears when token is required.

---

### Milestone 7 — Frontend: 2×2 Viewer Layout

**Goal**: Build the viewer page with 2×2 panel grid and expand/restore.

**Tasks**:

- [ ] **M7.1** Create `ViewerGrid2x2.tsx`:
  - CSS Grid: 2 columns × 2 rows, gap 2px, dark borders
  - Panels: [Axial, Sagittal] / [Coronal, 3D]
  - Each panel has a small label badge (top-left) with colored border matching
    crosshair convention (axial=yellow, sagittal=green, coronal=red, 3D=white)
- [ ] **M7.2** Create `ExpandablePanel.tsx`:
  - Wraps each panel content
  - Double-click header → expand to full area (CSS: `grid-column: 1/-1; grid-row: 1/-1`)
  - Double-click again or Esc → restore
  - Small expand icon button as alternative trigger
- [ ] **M7.3** Create `SeriesSelector.tsx`:
  - Fetch series list for current patient
  - Dropdown to switch series
  - Display phase badge + seg/voi indicators
- [ ] **M7.4** Wire `ViewerPage.tsx`:
  - Header: patient ID + series selector + layer controls (stubs)
  - Body: `ViewerGrid2x2` with four placeholder dark panels
  - Bottom bar: placeholder for sliders and W/L
- [ ] **M7.5** Verification:
  - Grid renders with correct proportions
  - Expand/restore works on each panel
  - Series dropdown populates from API

**Completion criteria**: 2×2 grid layout renders. Expand/restore works.
Series dropdown is functional. No viewer content yet (dark placeholders).

---

### Milestone 8 — Frontend: 2D Slice Viewers

**Goal**: Display interactive 2D slice views with slice navigation, crosshair
sync, W/L controls, and overlay toggles.

**Tasks**:

- [ ] **M8.1** Create `SliceView.tsx`:
  - Fetches PNG from `/api/slice/{axis}/{index}` and displays as `<img>`
  - Scroll wheel on panel → change slice index → re-fetch
  - Canvas crosshair overlay (two colored lines at current position)
  - Left-click → update crosshair position → notify parent
- [ ] **M8.2** Create `useSliceNavigation.ts` hook:
  - State: `{axial: number, coronal: number, sagittal: number}`
  - On crosshair click in one view, compute corresponding indices for others
  - Expose `setSlice(axis, index)` and `sliceIndices` to components
- [ ] **M8.3** Create `SliceSlider.tsx`:
  - Slider per axis below the grid (or inside each panel)
  - Bound to `useSliceNavigation` state
- [ ] **M8.4** Create `WindowLevelControl.tsx`:
  - Right-click drag on any 2D panel: horizontal=width, vertical=level
  - Numeric display of current W/L values
  - Preset buttons: Soft Tissue, Bone, Lung, Brain
- [ ] **M8.5** Create `useWindowLevel.ts` hook:
  - State: `{ww: number, wl: number}`
  - Presets: `{softTissue: [400, 50], bone: [1800, 400], lung: [1500, -600], brain: [80, 40]}`
- [ ] **M8.6** Create `LayerToggle.tsx` + `OpacitySlider.tsx`:
  - Per-label checkbox (kidney/tumor/cyst) with color indicator
  - Per-label opacity slider (0.0–1.0)
  - Changes trigger slice re-fetch with updated query params
- [ ] **M8.7** Wire everything into `ViewerPage`:
  - Load series on mount → fetch initial slices for all three axes
  - Crosshair sync between panels
  - W/L and overlay controls affect all slice fetches
- [ ] **M8.8** Verification:
  - Scrolling through axial slices shows correct anatomy
  - Clicking in axial view updates coronal + sagittal crosshairs
  - W/L drag changes contrast
  - Toggling kidney/tumor overlay shows/hides contours

**Completion criteria**: All three 2D panels display correct slices with working
crosshair sync, W/L adjustment, and overlay toggles. User can navigate the full
volume interactively.

---

### Milestone 9 — Frontend: 3D Surface Panel

**Goal**: Display 3D segmentation meshes in the fourth panel.

**Tasks**:

- [ ] **M9.1** Install `three` + `@react-three/fiber` + `@react-three/drei`
  (or `vtk.js` — choose the simpler option for mesh display).
- [ ] **M9.2** Create `Surface3DView.tsx`:
  - On series load, fetch GLB meshes for visible labels via `/api/mesh/{label}`
  - Render meshes with matching colors (cyan, yellow, magenta)
  - OrbitControls for rotate/pan/zoom
  - Dark background matching theme
- [ ] **M9.3** Create `BlendSlider.tsx`:
  - Global opacity slider (0.0–1.0) controlling mesh material transparency
- [ ] **M9.4** Handle empty state:
  - If no mask → dark panel with centered text "No segmentation available"
- [ ] **M9.5** Wire layer visibility toggles (FR-22) to also show/hide 3D meshes.
- [ ] **M9.6** Verification:
  - 3D panel shows kidney + tumor meshes for a seg-enabled series
  - Orbit/zoom works
  - Blend slider changes transparency
  - Toggling a layer hides it in both 2D and 3D
  - No-seg series shows empty panel with message

**Completion criteria**: 3D panel renders meshes with correct colors.
Layer toggles sync between 2D overlay and 3D meshes. Empty state handled.

---

### Milestone 10 — Frontend: Settings Persistence

**Goal**: Save and restore user preferences across sessions.

**Tasks**:

- [ ] **M10.1** Create `useSettings.ts` hook:
  - On viewer mount: fetch `GET /api/settings`, apply saved W/L, layers, last patient
  - On change: debounced `PUT /api/settings` (500ms delay)
- [ ] **M10.2** On `DatasetSelectorPage` load: if settings contain `last_patient`
  for the selected dataset, offer "Resume" button.
- [ ] **M10.3** Persist: W/L values, layer visibility, layer opacity, last patient ID,
  last series ID.
- [ ] **M10.4** Verification:
  - Change W/L, refresh browser → values restored
  - Navigate to patient, refresh → same patient loads

**Completion criteria**: All listed settings persist across browser refresh
and server restart.

---

### Milestone 11 — Integration Testing & Polish

**Goal**: End-to-end test the full workflow and fix issues.

**Tasks**:

- [ ] **M11.1** Test full flow: select dataset → patient → series → navigate → overlay → 3D → expand → restore
- [ ] **M11.2** Test VOI-only dataset (create or use Dataset820/voi/)
- [ ] **M11.3** Test NIfTI-only dataset (no seg/, no voi/)
- [ ] **M11.4** Test dataset without manifest.csv (filename-based discovery)
- [ ] **M11.5** Test auth flow: set token → verify 401 → login → session works
- [ ] **M11.6** Fix: responsiveness of 2×2 grid on different screen sizes
- [ ] **M11.7** Fix: loading indicators during slice fetch and mesh generation
- [ ] **M11.8** Fix: error messages for missing/corrupt files (graceful fallback)
- [ ] **M11.9** Performance: ensure slice navigation < 200ms, first render < 3s

**Completion criteria**: All acceptance criteria from SRS §10 pass.

---

### Milestone 12 — Containerization & Deployment

**Goal**: Build the single OCI image and verify it works with udocker.

**Tasks**:

- [ ] **M12.1** Create `Dockerfile`:
  - Multi-stage: build frontend → copy into Python image
  - FastAPI serves static files from `/app/static`
  - `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`
- [ ] **M12.2** Create `.dockerignore` (exclude `node_modules`, `__pycache__`, `.git`, data)
- [ ] **M12.3** Build and test with podman/docker:
  - `podman build -t radiology-ui:1.0 .`
  - `podman run -p 8000:8000 -v /path/to/data/dataset:/data:ro radiology-ui:1.0`
  - Verify all features work at `http://localhost:8000`
- [ ] **M12.4** Export and test with udocker:
  - `podman save -o radiology-ui_1.0.tar radiology-ui:1.0`
  - `udocker load -i radiology-ui_1.0.tar`
  - `udocker create --name=radio-ui radiology-ui:1.0`
  - `udocker run -p 8000:8000 -v /path/to/data:/data:ro radio-ui`
- [ ] **M12.5** Verify image size ≤ 1.5 GB.
- [ ] **M12.6** Update README with final deployment instructions.

**Completion criteria**: App runs from udocker with single load+create+run
sequence. All features work through the container. Image size within budget.

---

## Agent Instructions

### Before starting any milestone

1. Read **this file** (`AGENTS.md`) fully — especially the ⚠ Environment Constraints section.
2. Identify the current milestone: first one with any unchecked `[ ]` task.
3. Read `docs/SRS.md` sections relevant to this milestone.
4. Read the **Reference Files** listed above that apply to this milestone.
5. Check existing files in `backend/` and `frontend/` to understand current state before writing anything.

### While working on a milestone

6. Work on tasks **in order** (M0.1 → M0.2 → … never jump).
7. After completing each task, mark it `[x]` in this file immediately.
8. Run the verification steps listed in the milestone before declaring it done.
9. Do **not** start the next milestone without user confirmation.

### Environment rules (always enforced)

- ✅ Backend Python → always use `conda activate ccrcc` first
- ✅ Frontend / Node.js → always run inside `udocker` with `radio-node` container
- ✅ Data path → use `DATA_ROOT` env var defaulting to `../../data/dataset` (relative to `backend/`)
- ❌ Never use `sudo`, `apt-get`, `brew`, or system package managers
- ❌ Never run `npm`, `node`, or `npx` outside of a udocker container
- ❌ Never write files outside `radioccrcc-webui/`
- ❌ Never suggest SSH tunnel commands — VS Code Remote handles port forwarding
- ❌ Never create a Dockerfile or refer to Docker/Podman build until **M12**

### After completing a milestone

10. Mark all tasks `[x]` and update the **Progress Tracker** table.
11. Report what was completed, list the files created/modified, and state what M{N+1} will do.
12. **Stop and wait** for user confirmation before proceeding.

### Error handling

- If a task is blocked by a missing dependency or unclear requirement, **stop and ask** — do not guess.
- If a verification step fails, fix it within the current milestone before moving on.
- If a package is missing from Anaconda `ccrcc`, add it to `backend/requirements.txt` and run `pip install` inside the env.

---

## Progress Tracker

| Milestone | Title                              | Status         | Notes |
|-----------|------------------------------------|----------------|-------|
| M0        | Project Bootstrap & Dev Env        | ✅ Completed    | 2026-03-04 |
| M1        | Backend: Data Discovery Service    | ✅ Completed    | 2026-03-04 |
| M2        | Backend: Volume Loading & Slice    | ⬜ Not started  |       |
| M3        | Backend: 3D Mesh Generation        | ⬜ Not started  |       |
| M4        | Backend: Settings & Auth           | ⬜ Not started  |       |
| M5        | Frontend: Shell & Routing          | ⬜ Not started  |       |
| M6        | Frontend: Dataset & Patient Pages  | ⬜ Not started  |       |
| M7        | Frontend: 2×2 Viewer Layout        | ⬜ Not started  |       |
| M8        | Frontend: 2D Slice Viewers         | ⬜ Not started  |       |
| M9        | Frontend: 3D Surface Panel         | ⬜ Not started  |       |
| M10       | Frontend: Settings Persistence     | ⬜ Not started  |       |
| M11       | Integration Testing & Polish       | ⬜ Not started  |       |
| M12       | Containerization & Deployment      | ⬜ Not started  | Dockerfile created here only |
