# Software Requirements Specification (SRS)

## Radiology WebUI — Auxiliary Viewer Tool

| Field          | Value                                       |
|----------------|---------------------------------------------|
| **Version**    | 1.0                                         |
| **Date**       | 2026-03-04                                  |
| **Status**     | Approved for implementation                 |
| **Project**    | radio-ccrcc / Radiology WebUI               |
| **Author**     | Alonso (researcher) + GitHub Copilot (SRS)  |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Data Model & Folder Schema](#3-data-model--folder-schema)
4. [Functional Requirements](#4-functional-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [User Interface Specification](#6-user-interface-specification)
7. [Technical Architecture](#7-technical-architecture)
8. [Deployment](#8-deployment)
9. [Future Versions (Out of Scope v1.0)](#9-future-versions-out-of-scope-v10)
10. [Acceptance Criteria](#10-acceptance-criteria)
11. [Glossary](#11-glossary)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for a lightweight, local-first Web-based radiology viewer (codename **Radiology WebUI**) inspired by 3D Slicer. The tool is an **auxiliary utility** for researchers working with the `radio-ccrcc` pipeline, enabling interactive exploration of NIfTI volumes, nnU-Net segmentations, and VOI (Volume of Interest) crops on a per-patient basis.

### 1.2 Scope

**In scope (v1.0):**

- Browse a mounted dataset folder and discover patients, series, and masks automatically.
- Visualize NIfTI volumes and NumPy VOI arrays in a 2×2 layout: Axial, Coronal, Sagittal, and 3D surface rendering.
- Overlay multi-label segmentation masks (kidney/tumor/cyst) with per-layer visibility and opacity control.
- Expand any single panel to full view and restore to grid.
- Pan, zoom, window/level (W/L) adjustment, crosshair synchronization.
- Persist user preferences (last patient, W/L, opacity, visible layers).
- Simple token-based authentication for single-user local sessions.
- Run in a single container compatible with `udocker`.

**Out of scope (v1.0):**

- DICOM → NIfTI conversion (v2.0).
- Preprocessing / VOI extraction pipeline triggers (v2.0).
- Multi-series co-registration / fusion (v3.0).
- Clinical-grade certification or multi-user PACS workflow.

### 1.3 Target Users

Single researcher or engineer running the tool on a local machine or HPC node. The user is assumed to be familiar with the `radio-ccrcc` folder conventions.

### 1.4 References

| Reference                 | Description                                       |
|---------------------------|---------------------------------------------------|
| 3D Slicer (slicer.org)    | Open-source medical image viewer — UI inspiration |
| `nifti_visualizer.ipynb`  | Existing Jupyter NIfTI viewer in this project     |
| `voi_visualizer.ipynb`    | Existing Jupyter VOI viewer in this project       |
| `visualizer_utils.py`     | Shared rendering/overlay utilities                |
| `src/converter/`          | DICOM → NIfTI converter pipeline                  |
| `src/preprocessor/`       | VOI preprocessing pipeline                        |

---

## 2. Overall Description

### 2.1 Product Perspective

The WebUI replaces the Jupyter-based visualizers (`nifti_visualizer.ipynb`, `voi_visualizer.ipynb`) with a persistent, browser-accessible application that provides richer interaction (3D view, pan/zoom, expand panels) without requiring a running Jupyter kernel.

### 2.2 Operating Environment

| Aspect          | Specification                                                 |
|-----------------|---------------------------------------------------------------|
| Host OS         | Linux (primary), macOS (secondary)                            |
| Container       | OCI image run via `udocker`, `podman`, or `docker`            |
| Browser         | Chromium-based (Chrome, Edge, Brave) or Firefox, latest 2 ESR |
| Data mount      | Host path bind-mounted read-only at `/data` inside container  |
| Network         | `localhost` only; no external upload                          |

### 2.3 Constraints

- **Single container**: no Docker Compose or multi-service orchestration (udocker limitation).
- **CPU-only rendering**: no GPU passthrough required; 3D surface mesh computed server-side or via vtk.js client-side.
- **Read-only data**: the viewer never modifies, moves, or deletes files on the mounted path.

### 2.4 Assumptions

- Input data follows the folder schema defined in §3.
- Segmentation masks use integer labels: `0`=background, `1`=kidney, `2`=tumor, `3`=cyst.
- NIfTI files may not be in RAS orientation; the backend applies `nib.as_closest_canonical()` on load (same as existing notebooks).
- VOI NumPy arrays have shape `(X, Y, Z)` with the same axis convention as the notebooks.

---

## 3. Data Model & Folder Schema

The WebUI discovers data from a **Dataset root** (`DatasetID/`). Multiple datasets may coexist under a parent directory; the user selects which one to explore at runtime.

### 3.1 Canonical Dataset Layout

```
DatasetID/
├── manifest.csv                           # Patient/series metadata (from converter)
├── conversion_summary.json                # Converter run metadata
├── dataset.json                           # Preprocessor dataset summary
├── dataset_fingerprint.json               # Preprocessor fingerprint
├── patient_preprocess.csv                 # Preprocessor per-patient log
├── splits.json                            # Train/val/test splits
├── decisions.json                         # Viewer review decisions
│
├── nifti/                                 # Full CT volumes (flat)
│   ├── 01_case_00001_0000.nii.gz
│   ├── 02_case_00001_0000.nii.gz
│   └── ...
│
├── seg/                                   # nnU-Net segmentation masks (flat)
│   ├── 01_case_00001.nii.gz
│   ├── 02_case_00001.nii.gz
│   └── ...
│
├── nnunet/                                # nnU-Net preprocessed volumes (flat)
│   └── ...
│
└── voi/                                   # Cropped VOIs (hierarchical)
    ├── dataset.json
    ├── dataset_fingerprint.json
    ├── patient_preprocess.csv
    ├── splits.json
    │
    ├── images/{group}/{patient_id}/{phase}/
    │   └── NN_case_YYYYY_side{L,R}.npy
    │
    └── mask/{group}/{patient_id}/{phase}/
        └── NN_case_YYYYY_side{L,R}.npy
```

### 3.2 Naming Conventions

| Token           | Pattern              | Example               | Description                   |
|-----------------|----------------------|-----------------------|-------------------------------|
| `NN`            | `\d{2}`              | `01`                  | Per-case series counter       |
| `case_YYYYY`    | `case_\d{5}`         | `case_00042`          | Sequential case ID            |
| `_0000`         | literal              | `_0000`               | nnU-Net channel suffix (image)|
| `side{L,R}`     | `sideL` or `sideR`   | `sideL`               | Laterality (left/right kidney)|
| `{group}`       | string               | `A`, `B`, `NG`        | Patient classification group  |
| `{phase}`       | `NC\|ART\|VEN\|DELAY\|UNDEFINED` | `ART` | Contrast phase folder name    |

### 3.3 Mask Label Map (Multi-Label)

| Label | Structure   | Default color | Default alpha | Default visible |
|-------|-------------|---------------|---------------|-----------------|
| 0     | Background  | —             | —             | —               |
| 1     | Kidney      | cyan          | 0.15          | ✓               |
| 2     | Tumor       | yellow        | 0.20          | ✓               |
| 3     | Cyst        | magenta       | 0.15          | ✗               |

### 3.4 Patient-to-Series Mapping

One **patient** (`case_YYYYY`) may have:

- 1–N full NIfTI volumes in `nifti/` (multiple contrast phases / series).
- 0–N corresponding segmentation masks in `seg/` (matched by removing `_0000` suffix).
- 0–N VOI crops in `voi/images/` and `voi/mask/` (matched by group/patient_id/phase/filename).

The manifest.csv provides the mapping:

| Column            | Type   | Description                            |
|-------------------|--------|----------------------------------------|
| `filename`        | str    | NIfTI filename in `nifti/`             |
| `case_id`         | str    | `case_YYYYY`                           |
| `patient_id`      | str    | Original patient identifier            |
| `group`           | str    | Classification group (A/B/C/NG)        |
| `phase`           | str    | Contrast phase (nc/art/ven/delay/…)    |
| `protocol_source` | str    | How phase was determined               |

When `manifest.csv` is absent, the system falls back to filename-based discovery using the `case_YYYYY` regex pattern.

---

## 4. Functional Requirements

### 4.1 Data Discovery & Navigation

| ID       | Requirement                                                                                              | Priority  |
|----------|----------------------------------------------------------------------------------------------------------|-----------|
| FR-01    | **Dataset Selection**: User can browse and select a dataset root folder from the allowed mount path.     | Must      |
| FR-02    | **Patient Discovery**: System scans `nifti/`, `seg/`, and `voi/` to build a patient list grouped by `case_YYYYY`. If `manifest.csv` exists, enrich with group/phase metadata. | Must |
| FR-03    | **Series Discovery**: For a selected patient, list all available series (NIfTI volumes, segmentation masks, VOI crops) with their contrast phase and laterality. | Must |
| FR-04    | **Adaptive Content**: The viewer adapts to available data: (a) NIfTI-only → 2D MPR + empty 3D panel; (b) NIfTI + seg → 2D MPR with overlay + 3D surface; (c) VOI-only → 2D MPR from NumPy + overlay if mask exists + 3D surface if mask exists. | Must |
| FR-05    | **Patient Search/Filter**: User can search patients by ID, filter by group, or filter by contrast phase. | Should    |
| FR-06    | **Series Selector**: Sidebar or dropdown to switch between series of the same patient without returning to the patient list. | Must |

### 4.2 2D Visualization (MPR Views)

| ID       | Requirement                                                                                              | Priority  |
|----------|----------------------------------------------------------------------------------------------------------|-----------|
| FR-10    | **Axial View**: Display axial slice (`image[:, :, z].T`) with correct orientation.                       | Must      |
| FR-11    | **Coronal View**: Display coronal slice (`image[:, y, :].T`) with correct orientation.                   | Must      |
| FR-12    | **Sagittal View**: Display sagittal slice (`image[x, :, :].T`) with correct orientation.                 | Must      |
| FR-13    | **Slice Navigation**: Each view has a slider (or scroll) to navigate through slices.                      | Must      |
| FR-14    | **Crosshair Synchronization**: Selecting a point in one view updates the crosshair position in the other two views. Crosshair lines use distinct colors: axial=yellow, coronal=red, sagittal=green (matching existing notebooks). | Must |
| FR-15    | **Window/Level Control**: User can adjust HU window (width) and level (center) via drag interaction or numeric input. Default: W=400, L=50 (equivalent to HU_MIN=-150, HU_MAX=250). | Must |
| FR-16    | **W/L Presets**: Quick-select presets: Soft Tissue (W:400 L:50), Bone (W:1800 L:400), Lung (W:1500 L:-600), Brain (W:80 L:40). | Should |
| FR-17    | **Pan & Zoom**: User can pan and zoom within any 2D view. Zoom may optionally sync across views.         | Must      |
| FR-18    | **RAS Reorientation**: NIfTI volumes are reoriented to RAS on load for consistent display (using `nibabel.as_closest_canonical`). Original files are never modified. | Must |
| FR-19    | **NumPy VOI Support**: VOI arrays (`.npy` files) are loaded and displayed with the same axial/coronal/sagittal convention as the notebooks: shape `(X, Y, Z)`. | Must |

### 4.3 Segmentation Overlay

| ID       | Requirement                                                                                              | Priority  |
|----------|----------------------------------------------------------------------------------------------------------|-----------|
| FR-20    | **Mask Overlay**: If a segmentation mask (`seg/`) or VOI mask (`voi/mask/`) exists, overlay it on the 2D views. | Must |
| FR-21    | **Multi-Label Rendering**: Overlay renders labels 1 (kidney), 2 (tumor), 3 (cyst) with distinct colors and contour borders (matching `overlay_multi_layer_mask` behavior). | Must |
| FR-22    | **Per-Layer Visibility**: Each label has an independent visibility toggle (checkbox or button). Default: kidney=on, tumor=on, cyst=off. | Must |
| FR-23    | **Per-Layer Opacity**: Each label has an independent opacity slider (range 0.0–1.0). Default: kidney=0.15, tumor=0.20, cyst=0.15. | Must |
| FR-24    | **Mask Auto-Discovery**: System matches `seg/` masks to `nifti/` images by stripping the `_0000` suffix. For VOIs, masks are matched by identical path structure under `voi/mask/`. | Must |

### 4.4 3D Visualization

| ID       | Requirement                                                                                              | Priority  |
|----------|----------------------------------------------------------------------------------------------------------|-----------|
| FR-30    | **3D Surface Rendering**: When a segmentation mask exists, generate and display an isosurface mesh (marching cubes) for each visible label. | Must |
| FR-31    | **3D Blend Slider**: A global opacity/blend slider controls the transparency of the 3D surface rendering (range 0.0–1.0). | Must |
| FR-32    | **3D Rotate/Zoom**: User can orbit, pan, and zoom in the 3D viewport.                                    | Must      |
| FR-33    | **Empty 3D Panel**: When no mask is available, the 3D panel displays an empty dark canvas with an informational label ("No segmentation available"). Same behavior as 3D Slicer with no loaded segmentation. | Must |
| FR-34    | **Per-Label 3D Visibility**: Layer visibility toggles (FR-22) also apply to the 3D panel.                 | Should    |
| FR-35    | **3D Color Consistency**: Surface colors match the 2D overlay colors (cyan=kidney, yellow=tumor, magenta=cyst). | Must |

### 4.5 Layout & Expand

| ID       | Requirement                                                                                              | Priority  |
|----------|----------------------------------------------------------------------------------------------------------|-----------|
| FR-40    | **2×2 Grid Layout**: Default layout is a 2×2 grid: [Axial, Sagittal] / [Coronal, 3D].                   | Must      |
| FR-41    | **Expand Panel**: Double-click or button on any panel expands it to fill the full viewer area.             | Must      |
| FR-42    | **Restore Layout**: When expanded, a button or double-click restores the 2×2 grid.                        | Must      |
| FR-43    | **Resizable Panels**: Panels may be resized by dragging dividers (optional stretch goal).                  | Could     |

### 4.6 Persistence & Settings

| ID       | Requirement                                                                                              | Priority  |
|----------|----------------------------------------------------------------------------------------------------------|-----------|
| FR-50    | **Persist Last Patient**: Store last viewed patient ID per dataset so the user resumes where they left off. | Should |
| FR-51    | **Persist W/L Settings**: Save last-used window/level values per dataset.                                 | Should    |
| FR-52    | **Persist Layer Visibility**: Save layer on/off and opacity settings.                                     | Should    |
| FR-53    | **Settings Storage**: Preferences stored server-side as a JSON file (`<dataset>/viewer_settings.json`) or in browser `localStorage`. | Should |

### 4.7 Authentication

| ID       | Requirement                                                                                              | Priority  |
|----------|----------------------------------------------------------------------------------------------------------|-----------|
| FR-60    | **Simple Auth**: A single shared token or password protects the web interface. Configurable via environment variable (`RADIOLOGY_UI_TOKEN`). | Should |
| FR-61    | **Session Persistence**: Once authenticated, the session persists until browser close or explicit logout.  | Should    |

---

## 5. Non-Functional Requirements

| ID       | Requirement                                                                                              | Target     |
|----------|----------------------------------------------------------------------------------------------------------|------------|
| NFR-01   | **First Slice Visible**: After selecting a patient/series, the first 2D slice renders in < 3 seconds for a typical volume (512×512×200). | < 3 s |
| NFR-02   | **Slice Navigation Latency**: Changing a single slice (slider drag) updates the view in < 200 ms.         | < 200 ms   |
| NFR-03   | **3D Mesh Generation**: Initial marching-cubes computation completes in < 5 seconds for typical segmentation size. | < 5 s |
| NFR-04   | **Memory Footprint**: Backend holds at most 2 volumes in memory simultaneously (current + preloaded next). | ≤ 2 volumes |
| NFR-05   | **Container Image Size**: Final OCI image ≤ 1.5 GB (Python runtime + Node build + dependencies).          | ≤ 1.5 GB   |
| NFR-06   | **Read-Only Data**: Application never writes to the mounted dataset path. Settings stored separately.      | Mandatory  |
| NFR-07   | **Privacy / Local-Only**: No data leaves the host. No telemetry, analytics, or external API calls.         | Mandatory  |
| NFR-08   | **Browser Compatibility**: Chrome 100+, Firefox 100+, Edge 100+.                                           | Must       |
| NFR-09   | **Maintainability**: Backend services are modular (one file per concern). Frontend uses typed TypeScript with component isolation. | Must |
| NFR-10   | **Logging**: Backend logs requests and errors to stdout (container-friendly). Log level configurable via env var. | Should |

---

## 6. User Interface Specification

### 6.1 Page Flow

```
┌──────────────────────────────────────────────────────────┐
│  [1] Dataset Selector Page                                │
│   • Browse allowed folders under /data                    │
│   • Show dataset summary (patient count, has seg, etc.)   │
│   • Click dataset → Patient List                          │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  [2] Patient List Page                                     │
│   • Table: patient_id | group | #series | #seg | #voi     │
│   • Search bar + group/phase filters                       │
│   • Click patient → Viewer                                 │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  [3] Viewer Page (main workspace)                          │
│   • Header: patient ID, series selector, layer controls    │
│   • 2×2 grid: Axial | Sagittal | Coronal | 3D              │
│   • Bottom bar: slice sliders, W/L controls                 │
│   • Expand/restore per panel                                │
└──────────────────────────────────────────────────────────┘
```

### 6.2 Viewer Page Wireframe (2×2 Layout)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Patient: case_00042  │ Series: [▼ 01_ART ▼]  │ 🟢Kidney 🟡Tumor   │
├────────────────────────────┬────────────────────────────────────────┤
│                            │                                        │
│        AXIAL               │        SAGITTAL                        │
│    (yellow border)         │      (green border)                    │
│                            │                                        │
│   [scroll / slider Z]      │   [scroll / slider X]                  │
│   [⤢ expand]               │   [⤢ expand]                          │
│                            │                                        │
├────────────────────────────┼────────────────────────────────────────┤
│                            │                                        │
│        CORONAL             │         3D SURFACE                     │
│     (red border)           │      (orbit / zoom)                    │
│                            │                                        │
│   [scroll / slider Y]      │   [blend slider 0–1]                  │
│   [⤢ expand]               │   [⤢ expand]                          │
│                            │                                        │
├────────────────────────────┴────────────────────────────────────────┤
│  W/L: [drag area]  Presets: [Soft Tissue] [Bone] [Lung] [Brain]    │
│  Opacity: Kidney [——●———] 0.15   Tumor [———●——] 0.20               │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 Color Scheme

Dark theme (inspired by 3D Slicer dark mode):

| Element             | Color      |
|---------------------|------------|
| Background          | `#1e1e1e`  |
| Panel background    | `#121212`  |
| Panel borders       | `#333`     |
| Text primary        | `#ddd`     |
| Text secondary      | `#aaa`     |
| Axial crosshair     | `yellow`   |
| Coronal crosshair   | `red`      |
| Sagittal crosshair  | `green`    |

### 6.4 Interaction Summary

| Action                  | Input                      | Effect                                   |
|-------------------------|----------------------------|------------------------------------------|
| Navigate slice          | Scroll wheel / slider      | Update slice index in that view           |
| Adjust W/L              | Right-click drag on 2D     | Horizontal=width, vertical=level          |
| Pan                     | Middle-click drag (or Shift+left) | Move viewport origin              |
| Zoom                    | Ctrl+scroll / pinch        | Zoom in/out centered on cursor            |
| Crosshair click         | Left-click on 2D view      | Sets crosshair position, syncs all views  |
| Expand panel            | Double-click header        | Panel fills full viewer area              |
| Restore grid            | Double-click or Esc        | Return to 2×2 layout                     |
| Orbit 3D                | Left-click drag on 3D      | Rotate camera around subject              |
| Zoom 3D                 | Scroll on 3D panel         | Dolly camera in/out                       |
| Toggle layer            | Click checkbox             | Show/hide label in 2D overlay + 3D mesh   |
| Change opacity          | Drag slider                | Adjust alpha for selected label            |

---

## 7. Technical Architecture

### 7.1 Stack Selection

| Layer        | Technology                  | Rationale                                            |
|--------------|-----------------------------|------------------------------------------------------|
| Backend      | **Python 3.11 + FastAPI**   | Native nibabel/numpy; async; lightweight             |
| Frontend     | **React 18 + TypeScript**   | Type safety; component model; rich ecosystem         |
| 2D Rendering | **Cornerstone3D** (or **Niivue**) | Purpose-built medical image 2D/MPR viewer     |
| 3D Rendering | **vtk.js** (via Cornerstone3D) or **three.js** | Isosurface rendering in browser     |
| Bundler      | **Vite**                    | Fast builds; works well with TypeScript + React      |
| Container    | **OCI / Docker**            | Single-stage build; udocker-compatible               |

### 7.2 Backend Architecture

```
backend/
├── app/
│   ├── main.py                # FastAPI app, static file serving, CORS, auth
│   ├── config.py              # Settings from env vars (token, log level, paths)
│   │
│   ├── api/
│   │   ├── datasets.py        # GET /api/datasets — list available datasets
│   │   ├── patients.py        # GET /api/datasets/{id}/patients
│   │   ├── series.py          # GET /api/datasets/{id}/patients/{pid}/series
│   │   ├── slices.py          # GET /api/slices/{axis}/{index} — returns PNG
│   │   ├── mesh.py            # GET /api/mesh/{label} — returns GLB/OBJ
│   │   └── settings.py        # GET/PUT /api/settings
│   │
│   ├── services/
│   │   ├── discovery.py       # Folder scanning, manifest parsing, patient grouping
│   │   ├── nifti_loader.py    # nibabel load + RAS reorientation + HU normalize
│   │   ├── numpy_loader.py    # .npy VOI load
│   │   ├── mask_loader.py     # Segmentation mask load + label extraction
│   │   ├── slice_renderer.py  # Extract 2D slice as PNG (with overlay)
│   │   ├── mesh_generator.py  # Marching cubes → GLB/OBJ mesh
│   │   └── settings_store.py  # JSON-based persistence
│   │
│   └── models/
│       ├── patient.py         # Pydantic models for API responses
│       └── dataset.py         # Dataset and series models
│
└── requirements.txt
```

**Key backend design decisions:**

- **Slice-as-PNG API**: The backend renders 2D slices server-side as PNG images (grayscale + optional overlay). This avoids transferring raw float arrays to the browser and reduces frontend complexity. Alternative: raw array transfer + client-side rendering (may be adopted in v2.0 for smoother interaction).
- **Mesh-as-GLB API**: 3D meshes are generated server-side via `skimage.measure.marching_cubes` and served as binary GLB files. The frontend loads them into vtk.js or three.js.
- **Volume cache**: Keep the current volume + segmentation in memory. Evict on patient/series change.

### 7.3 Frontend Architecture

```
frontend/
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   │
│   ├── pages/
│   │   ├── DatasetSelectorPage.tsx
│   │   ├── PatientListPage.tsx
│   │   └── ViewerPage.tsx
│   │
│   ├── components/
│   │   ├── layout/
│   │   │   ├── ViewerGrid2x2.tsx        # 2×2 panel grid with expand/restore
│   │   │   └── ExpandablePanel.tsx      # Single panel with expand button
│   │   │
│   │   ├── viewers/
│   │   │   ├── SliceView.tsx            # 2D slice display (Axial/Cor/Sag)
│   │   │   ├── Surface3DView.tsx        # 3D mesh viewer
│   │   │   └── CrosshairOverlay.tsx     # Synchronized crosshair lines
│   │   │
│   │   ├── controls/
│   │   │   ├── SliceSlider.tsx          # Slice navigation slider
│   │   │   ├── WindowLevelControl.tsx   # W/L drag control + presets
│   │   │   ├── LayerToggle.tsx          # Per-label visibility checkbox
│   │   │   ├── OpacitySlider.tsx        # Per-label opacity slider
│   │   │   ├── BlendSlider.tsx          # 3D blend/opacity slider
│   │   │   └── SeriesSelector.tsx       # Dropdown for patient series
│   │   │
│   │   └── navigation/
│   │       ├── DatasetBrowser.tsx       # Folder tree / card selector
│   │       ├── PatientTable.tsx         # Sortable/filterable patient list
│   │       └── SearchBar.tsx            # Patient ID search + filters
│   │
│   ├── services/
│   │   └── api.ts                       # Typed fetch wrappers for all endpoints
│   │
│   ├── hooks/
│   │   ├── useSliceNavigation.ts
│   │   ├── useWindowLevel.ts
│   │   └── useSettings.ts
│   │
│   ├── types/
│   │   └── index.ts                     # Shared TypeScript interfaces
│   │
│   └── styles/
│       └── theme.ts                     # Dark theme tokens
│
├── package.json
├── tsconfig.json
└── vite.config.ts
```

### 7.4 API Contract (Summary)

| Method | Endpoint                                                 | Returns                  | Description                                 |
|--------|----------------------------------------------------------|--------------------------|---------------------------------------------|
| GET    | `/api/datasets`                                          | `Dataset[]`              | List available dataset folders              |
| GET    | `/api/datasets/{dsid}/patients`                          | `Patient[]`              | List patients in dataset                    |
| GET    | `/api/datasets/{dsid}/patients/{pid}/series`             | `Series[]`               | List series for patient                     |
| POST   | `/api/datasets/{dsid}/patients/{pid}/series/{sid}/load`  | `VolumeInfo`             | Load volume into memory, return shape/spacing |
| GET    | `/api/slice/{axis}/{index}?ww=400&wl=50&layers=1,2`     | `image/png`              | Render 2D slice as PNG                      |
| GET    | `/api/mesh/{label}?smooth=true`                          | `model/gltf-binary`      | Get 3D surface mesh                         |
| GET    | `/api/settings`                                          | `Settings`               | Get persisted user preferences              |
| PUT    | `/api/settings`                                          | `Settings`               | Save user preferences                       |

### 7.5 Data Flow

```
[Browser]                          [FastAPI Backend]              [Filesystem]
    │                                     │                            │
    │  GET /api/datasets                  │                            │
    │ ──────────────────────────────────► │  scan /data/*/             │
    │                                     │ ──────────────────────────►│
    │  ◄────── Dataset[]                  │  ◄─── folder list          │
    │                                     │                            │
    │  GET /api/.../patients              │                            │
    │ ──────────────────────────────────► │  parse manifest.csv        │
    │                                     │  + scan nifti/, seg/, voi/ │
    │  ◄────── Patient[]                  │                            │
    │                                     │                            │
    │  POST /api/.../load                 │                            │
    │ ──────────────────────────────────► │  nib.load() → RAS → cache │
    │  ◄────── VolumeInfo{shape,spacing}  │                            │
    │                                     │                            │
    │  GET /api/slice/axial/120           │                            │
    │ ──────────────────────────────────► │  extract slice → PNG       │
    │  ◄────── image/png                  │                            │
    │                                     │                            │
    │  GET /api/mesh/2                    │                            │
    │ ──────────────────────────────────► │  marching_cubes → GLB      │
    │  ◄────── model/gltf-binary          │                            │
```

---

## 8. Deployment

### 8.1 Dockerfile (Single-Stage Concept)

```dockerfile
FROM python:3.11-slim AS backend
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .

FROM node:20-alpine AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY --from=backend /app /app
COPY --from=frontend /web/dist /app/static
EXPOSE 8000
ENV RADIOLOGY_UI_TOKEN=""
ENV LOG_LEVEL="info"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2 udocker Run Command

```bash
# 1. Load image
udocker load -i radiology-ui_1.0.tar

# 2. Create container
udocker create --name=radio-ui radiology-ui:1.0

# 3. Run with mounted dataset
udocker run \
  -p 8000:8000 \
  -v /path/to/data/dataset:/data:ro \
  -e RADIOLOGY_UI_TOKEN=mysecret \
  radio-ui
```

### 8.3 Configuration (Environment Variables)

| Variable              | Default         | Description                                    |
|-----------------------|-----------------|------------------------------------------------|
| `RADIOLOGY_UI_TOKEN`  | `""` (no auth)  | Token for simple authentication                |
| `DATA_ROOT`           | `/data`         | Mount point for dataset files                  |
| `LOG_LEVEL`           | `info`          | Python logging level                           |
| `HOST`                | `0.0.0.0`       | Bind address                                   |
| `PORT`                | `8000`          | Bind port                                      |

---

## 9. Future Versions (Out of Scope v1.0)

| Version | Feature                                                                           |
|---------|-----------------------------------------------------------------------------------|
| v2.0    | DICOM → NIfTI conversion trigger from UI (calls `src/converter/`)                 |
| v2.0    | VOI preprocessing trigger from UI (calls `src/preprocessor/`)                     |
| v2.0    | Raw array streaming + client-side rendering for smoother slice scrolling           |
| v2.0    | Annotation / labeling tools (draw ROI, measure distance)                          |
| v3.0    | Multi-series co-registration / overlay (different phases fused)                   |
| v3.0    | Volumetric rendering (ray casting) in addition to surface mesh                    |
| v3.0    | Multi-user sessions with role-based access                                        |

---

## 10. Acceptance Criteria

| #  | Criterion                                                                                              | Verified by        |
|----|--------------------------------------------------------------------------------------------------------|--------------------|
| 1  | User selects a dataset folder and sees the patient list within 5 seconds.                              | Manual test        |
| 2  | User selects a patient and sees 2×2 view with correct axial/coronal/sagittal orientation.              | Manual test        |
| 3  | Scrolling a slice slider updates the corresponding 2D view in < 200 ms.                                | Manual test        |
| 4  | Crosshair click in one panel updates the other two.                                                    | Manual test        |
| 5  | W/L drag adjusts intensity in real time.                                                               | Manual test        |
| 6  | If segmentation mask exists, overlay appears with contour borders; toggling layers works.              | Manual test        |
| 7  | If segmentation mask exists, 3D panel shows surface mesh; blend slider controls transparency.          | Manual test        |
| 8  | If no mask exists, 3D panel shows dark canvas with "No segmentation available" label.                  | Manual test        |
| 9  | Double-click on any panel expands it; double-click again restores 2×2.                                 | Manual test        |
| 10 | VOI `.npy` series are loadable and displayable with the same interaction as NIfTI series.              | Manual test        |
| 11 | Application runs via `udocker run` with a single `docker load` + `create` + `run` sequence.           | Deployment test    |
| 12 | No files are modified on the mounted dataset path (`/data`).                                           | Audit / strace     |
| 13 | Setting `RADIOLOGY_UI_TOKEN` blocks unauthenticated access.                                            | Manual test        |
| 14 | Last-viewed patient and W/L settings persist across browser refresh.                                   | Manual test        |

---

## 11. Glossary

| Term          | Definition                                                                |
|---------------|---------------------------------------------------------------------------|
| **MPR**       | Multi-Planar Reconstruction — axial, coronal, sagittal slice views       |
| **W/L**       | Window Width / Window Level — intensity windowing for CT display         |
| **HU**        | Hounsfield Units — standard CT intensity scale                           |
| **RAS**       | Right-Anterior-Superior — standard neuroimaging orientation convention   |
| **VOI**       | Volume of Interest — cropped and resampled sub-volume around a structure |
| **nnU-Net**   | Self-configuring framework for medical image segmentation                |
| **NIfTI**     | Neuroimaging Informatics Technology Initiative file format (`.nii.gz`)   |
| **GLB**       | Binary glTF — compact 3D mesh format                                    |
| **OCI**       | Open Container Initiative — container image specification                |
| **udocker**   | User-space container execution tool (no root required)                   |
| **Marching Cubes** | Algorithm to extract isosurface mesh from a 3D scalar field        |

---

*End of SRS v1.0*
