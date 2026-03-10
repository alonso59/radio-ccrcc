# Software Requirements Specification (SRS)

## Radiology WebUI — Auxiliary Viewer Tool

| Field          | Value                                       |
|----------------|---------------------------------------------|
| **Version**    | 1.2                                         |
| **Date**       | 2026-03-05                                  |
| **Status**     | Baseline implemented (living specification) |
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
- Reviewer workflow in Viewer: patient/group navigator, next-patient loading,
  staged phase reclassification (`NC/ART/VEN`), staged delete-to-recycle, and
  apply-with-confirmation with audit logs.
- Run as one web service image (backend + compiled frontend), compatible with `docker`, `podman`, and `udocker`.

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
| Data mount      | Host path bind-mounted at `/data`; write access required only for review apply operations |
| Network         | `localhost` only; no external upload                          |

### 2.3 Constraints

- **Single service runtime**: backend API and compiled frontend are served by one container process.
- **Compose optional**: native Docker/Podman can use `docker-compose.yml` for convenience; `udocker` can run the same OCI image without compose.
- **CPU-only rendering**: no GPU passthrough required; 3D surface mesh is generated server-side and rendered client-side with three.js.
- **Controlled data mutations**: only the explicit review-apply workflow may
  move/update dataset files, and only when `ALLOW_DATA_MUTATIONS=true`.

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
├── decisions.json                         # Viewer review audit log (append-only)
├── reclassification_log.json              # Batch reclassification report
├── deletion_log.json                      # Batch deletion report
├── deleted/                               # Recycle bin for NIfTI/SEG file moves
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
    └── deleted/                           # Recycle bin for VOI image/mask moves
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
| FR-33    | **Empty 3D Panel**: When no mask is available, the 3D panel displays an informational empty-state view ("No segmentation available") in the same panel slot. | Must |
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
| FR-53    | **Settings Storage**: Preferences are persisted server-side in a JSON file (default `~/.radiology-webui/settings.json`, fallback `/tmp/radiology-webui/settings.json`) keyed by dataset ID. | Should |

### 4.7 Authentication

| ID       | Requirement                                                                                              | Priority  |
|----------|----------------------------------------------------------------------------------------------------------|-----------|
| FR-60    | **Simple Auth**: A single shared bearer token protects API routes (`/api/*`, excluding `/api/health`). Token is configured via environment variable (`RADIOLOGY_UI_TOKEN`). | Should |
| FR-61    | **Session Persistence**: Auth token is memory-only by default; optional browser persistence via `VITE_AUTH_TOKEN_STORAGE=local`. No dedicated logout control is required in v1.0. | Should |

### 4.8 Reviewer Workflow (Viewer Page)

| ID       | Requirement                                                                                              | Priority  |
|----------|----------------------------------------------------------------------------------------------------------|-----------|
| FR-70    | **Viewer Patient Selector**: Viewer page provides a patient dropdown (`case_YYYYY`) for direct patient switching inside the dataset context. | Must |
| FR-71    | **Viewer Group Filter**: Viewer page provides a group filter with `All` plus dynamic dataset groups (`A`, `B`, `NG`, etc.). | Must |
| FR-72    | **Next Patient Button**: Viewer page provides `Load Next Patient` that follows sorted patient order within the active group filter. | Must |
| FR-73    | **Phase Decision Control**: Viewer page provides a phase decision selector limited to `NC`, `ART`, `VEN` for the currently selected series. | Must |
| FR-74    | **Delete Decision Control**: Viewer page provides delete decision for the currently selected series using recycle-bin semantics (never hard delete). | Must |
| FR-75    | **Staged Queue + Apply**: Review actions are staged client-side (current patient scope) and only executed after explicit confirmation in `Apply Changes`. | Must |
| FR-76    | **Apply Endpoint**: `POST /api/datasets/{dataset_id}/review/apply` executes queued operations and returns per-operation status (`applied`, `skipped`, `failed`) plus batch summary. | Must |
| FR-77    | **Mutation Safety Gate**: Review apply is rejected with `409` unless `ALLOW_DATA_MUTATIONS=true`. | Must |
| FR-78    | **Audit Artifacts**: Applying review actions appends records to `<dataset>/decisions.json` and updates batch logs `<dataset>/reclassification_log.json`, `<dataset>/deletion_log.json`. | Must |
| FR-79    | **NIfTI Reclassify Rule**: For NIfTI series, reclassification updates `manifest.csv` (`phase`, `protocol_source`) if present and does not move NIfTI files. | Must |
| FR-80    | **Recycle Paths**: NIfTI/SEG delete moves files under `<dataset>/deleted/...`; VOI delete moves files under `<dataset>/voi/deleted/...`. | Must |

---

## 5. Non-Functional Requirements

| ID       | Requirement                                                                                              | Target     |
|----------|----------------------------------------------------------------------------------------------------------|------------|
| NFR-01   | **First Slice Visible**: After selecting a patient/series, the first 2D slice renders in < 3 seconds for a typical volume (512×512×200). | < 3 s |
| NFR-02   | **Slice Navigation Latency**: Changing a single slice (slider drag) updates the view in < 200 ms.         | < 200 ms   |
| NFR-03   | **3D Mesh Generation**: Initial marching-cubes computation completes in < 5 seconds for typical segmentation size. | < 5 s |
| NFR-04   | **Memory Footprint**: Backend holds at most 2 volumes in memory simultaneously (current + preloaded next). | ≤ 2 volumes |
| NFR-05   | **Container Image Size**: Final OCI image ≤ 1.5 GB (Python runtime + Node build + dependencies).          | ≤ 1.5 GB   |
| NFR-06   | **Controlled Mutations Only**: Dataset writes are allowed only through confirmed review apply operations (phase reclassify/delete recycle) with audit logs. | Mandatory  |
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
│   • Header: group filter, patient selector, next-patient   │
│   • Series selector + review queue controls (phase/delete) │
│   • Apply changes confirmation for staged review actions   │
│   • 2×2 grid: Axial | Sagittal | Coronal | 3D              │
│   • Bottom bar: slice sliders, W/L controls                 │
│   • Expand/restore per panel                                │
└──────────────────────────────────────────────────────────┘
```

### 6.2 Viewer Page Wireframe (2×2 Layout)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Group: [▼ All ▼]  Patient: [▼ case_00042 ▼]  [Load Next Patient]   │
│ Series: [▼ 01_case_00042_0000 ▼]  Decision: [▼ NC/ART/VEN ▼]       │
│ [Queue Reclassify] [Queue Delete] [Clear Pending] [Apply Changes]   │
│                                                       🟢Kidney 🟡Tumor │
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
| Zoom                    | Ctrl/Cmd+scroll on panel   | Zoom in/out centered on cursor; prevents browser page zoom in-panel |
| Reset view fit          | `Fit` button / double-click slice viewport | Reset panel pan+zoom to fitted defaults |
| Crosshair click         | Left-click on 2D view      | Sets crosshair position, syncs all views  |
| Expand panel            | Expand button / header double-click | Panel fills full viewer area       |
| Restore grid            | Restore button / header double-click / Esc | Return to 2×2 layout             |
| Orbit 3D                | Left-click drag on 3D      | Rotate camera around subject              |
| Zoom 3D                 | Scroll on 3D panel         | Dolly camera in/out                       |
| Toggle layer            | Click checkbox             | Show/hide label in 2D overlay + 3D mesh   |
| Change opacity          | Drag slider                | Adjust alpha for selected label            |
| Filter by group         | Group dropdown             | Restrict viewer patient dropdown + next sequence |
| Switch patient          | Patient dropdown           | Navigate to selected patient without leaving viewer |
| Load next patient       | Button click               | Navigate to next filtered patient          |
| Stage reclassification  | Phase selector + queue button | Add `NC/ART/VEN` action for selected series |
| Stage delete            | Queue delete button        | Add delete action for selected series      |
| Apply staged actions    | Apply button + confirmation| Execute filesystem/manifest updates and write audit logs |

---

## 7. Technical Architecture

### 7.1 Stack Selection

| Layer        | Technology                  | Rationale                                            |
|--------------|-----------------------------|------------------------------------------------------|
| Backend      | **Python 3.12 + FastAPI**   | Native nibabel/numpy stack; lightweight API + static serving |
| Frontend     | **React 19 + TypeScript + MUI** | Typed UI and reusable component primitives       |
| 2D Rendering | **Server-rendered PNG slices + React interaction layer** | Notebook-aligned render pipeline with simple browser display |
| 3D Rendering | **three.js via @react-three/fiber + GLTFLoader** | Interactive rendering of server-generated GLB meshes |
| Bundler      | **Vite 7**                  | Fast TypeScript builds and dev-server proxy          |
| Container    | **Multi-stage OCI image**   | Node build stage + Python runtime stage, non-root runtime user |

### 7.2 Backend Architecture

```
backend/
├── app/
│   ├── main.py                # FastAPI app, router mounting, static SPA serving
│   ├── config.py              # Settings from env vars (token, log level, paths)
│   │
│   ├── middleware/
│   │   └── auth.py            # Bearer auth middleware for /api/*
│   │
│   ├── api/
│   │   ├── datasets.py        # GET /api/datasets — list available datasets
│   │   ├── slices.py          # POST load + GET /api/slice/{axis}/{index}
│   │   ├── mesh.py            # GET /api/mesh/{label} — returns GLB/OBJ
│   │   ├── review.py          # POST /api/datasets/{id}/review/apply
│   │   └── settings.py        # GET/PUT /api/settings
│   │
│   ├── services/
│   │   ├── discovery.py       # Folder scanning, manifest parsing, patient grouping
│   │   ├── nifti_loader.py    # nibabel load + RAS reorientation + HU normalize
│   │   ├── numpy_loader.py    # .npy VOI load
│   │   ├── mask_loader.py     # Segmentation mask load + label extraction
│   │   ├── slice_renderer.py  # Extract 2D slice as PNG (with overlay)
│   │   ├── mesh_generator.py  # Marching cubes → GLB/OBJ mesh
│   │   ├── settings_store.py  # JSON-based persistence
│   │   ├── review_apply.py    # Reclassify/delete apply + audit logs
│   │   └── volume_cache.py    # LRU series cache + expiring load handles
│   │
│   └── models/
│       ├── dataset.py         # Dataset/patient/series/volume response models
│       ├── review.py          # Review operation and batch result models
│       └── settings.py        # Per-dataset viewer settings payload
│
└── requirements.txt
```

**Key backend design decisions:**

- **Slice-as-PNG API**: The backend renders 2D slices server-side as PNG images (grayscale + optional overlay), keeping notebook-equivalent orientation and label rendering.
- **Load handle contract**: Series load returns `load_handle`; slice and mesh calls require it so stale/expired cache state can be rejected explicitly (`410`).
- **Mesh-as-GLB API**: 3D meshes are generated server-side via `skimage.measure.marching_cubes` and served as binary GLB files.
- **Volume cache**: In-memory LRU cache for up to 2 volumes plus expiring handle registry for safe revalidation.
- **Review apply gate**: dataset mutation endpoints are blocked unless `ALLOW_DATA_MUTATIONS=true`.
- **Audit-first mutation flow**: phase/delete operations append audit events and keep deleted files recoverable in recycle paths.

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
│   │   ├── LoginDialog.tsx              # Token prompt for 401 retry flow
│   │   └── viewer/
│   │       ├── ViewerGrid2x2.tsx        # 2×2 panel grid with expand/restore
│   │       ├── ExpandablePanel.tsx      # Panel chrome with expand/restore controls
│   │       ├── SliceView.tsx            # 2D slice display (Axial/Cor/Sag)
│   │       ├── Surface3DView.tsx        # 3D mesh viewer
│   │       ├── SliceSlider.tsx          # Slice navigation slider
│   │       ├── WindowLevelControl.tsx   # W/L presets
│   │       ├── LayerToggle.tsx          # Per-label visibility checkbox
│   │       ├── OpacitySlider.tsx        # Per-label opacity slider
│   │       ├── BlendSlider.tsx          # 3D blend/opacity slider
│   │       ├── SeriesSelector.tsx       # Dropdown for patient series
│   │       ├── useSliceNavigation.ts    # Crosshair/slice synchronization state
│   │       └── useWindowLevel.ts        # W/L state + presets
│   │
│   ├── hooks/
│   │   └── useSettings.ts               # Persisted settings hydration/sync
│   │
│   ├── services/
│   │   └── api.ts                       # Typed API client + auth retry interceptors
│   │
│   └── styles/
│       └── theme.ts                     # Dark theme tokens
│
├── package.json
├── tsconfig.json
└── vite.config.ts
```

**Key frontend design decisions:**

- **Stateful API wrapper**: Axios interceptors manage bearer token injection and 401 re-prompt flow.
- **Load-handle aware rendering**: Viewer data requests are tied to `load_handle` so stale series state can recover by reloading.
- **Smooth refresh behavior**: 2D/3D panels keep previous content during transient refresh failures where possible, reducing visible flicker.

### 7.4 API Contract (Summary)

| Method | Endpoint                                                                              | Returns                  | Description                                 |
|--------|---------------------------------------------------------------------------------------|--------------------------|---------------------------------------------|
| GET    | `/api/health`                                                                         | `{"status":"ok"}`        | Backend health check                        |
| GET    | `/api/datasets`                                                                       | `Dataset[]`              | List available dataset folders              |
| GET    | `/api/datasets/{dataset_id}/patients`                                                 | `Patient[]`              | List patients in dataset                    |
| GET    | `/api/datasets/{dataset_id}/patients/{patient_id}/series`                             | `Series[]`               | List series for patient                     |
| POST   | `/api/datasets/{dataset_id}/patients/{patient_id}/series/{series_id}/load`            | `VolumeInfo`             | Load volume into cache and return `load_handle`, shape, spacing, labels |
| GET    | `/api/slice/{axis}/{index}?load_handle=...&ww=...&wl=...&layers=1,2&opacity_1=...`   | `image/png`              | Render 2D slice from cached series          |
| GET    | `/api/mesh/{label}?load_handle=...&smooth=true`                                       | `model/gltf-binary`      | Get 3D surface mesh for one label           |
| POST   | `/api/datasets/{dataset_id}/review/apply`                                              | `ReviewApplyResponse`    | Apply staged reclassify/delete operations with audit logging |
| GET    | `/api/settings`                                                                       | `Settings`               | Get persisted dataset-scoped viewer prefs   |
| PUT    | `/api/settings`                                                                       | `Settings`               | Save persisted dataset-scoped viewer prefs  |

### 7.5 Data Flow

```
[Browser]                          [FastAPI Backend]                    [Filesystem]
    │                                     │                                  │
    │  GET /api/datasets                  │                                  │
    │ ──────────────────────────────────► │  scan DATA_ROOT/Dataset*/        │
    │                                     │ ─────────────────────────────────►│
    │  ◄────── Dataset[]                  │  ◄─── folder list                │
    │                                     │                                  │
    │  GET /api/.../patients              │                                  │
    │ ──────────────────────────────────► │  parse manifest.csv + scan trees │
    │  ◄────── Patient[]                  │                                  │
    │                                     │                                  │
    │  POST /api/.../load                 │                                  │
    │ ──────────────────────────────────► │  load volume/mask → cache        │
    │  ◄────── VolumeInfo{load_handle,...}│                                  │
    │                                     │                                  │
    │  GET /api/slice/... ?load_handle=H  │                                  │
    │ ──────────────────────────────────► │  cache lookup + render PNG       │
    │  ◄────── image/png                  │                                  │
    │                                     │                                  │
    │  GET /api/mesh/... ?load_handle=H   │                                  │
    │ ──────────────────────────────────► │  marching_cubes → GLB            │
    │  ◄────── model/gltf-binary          │                                  │
    │                                     │                                  │
    │  POST /api/.../review/apply         │                                  │
    │ ──────────────────────────────────► │  validate + apply batch ops      │
    │                                     │  (manifest update / file moves)  │
    │                                     │ ─────────────────────────────────►│
    │  ◄────── batch summary + per-op     │  append decisions + logs         │
```

---

## 8. Deployment

### 8.1 Dockerfile (Implemented Multi-Stage Image)

```dockerfile
FROM node:20-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_ROOT=/data
WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN addgroup --system app && adduser --system --ingroup app app
COPY --chown=app:app backend /app/backend
COPY --chown=app:app --from=frontend-build /build/frontend/dist /app/static
EXPOSE 8000
WORKDIR /app/backend
HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status==200 else 1)"
USER app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2 Native Docker/Podman (Recommended)

```bash
cp .env.example .env
# edit DATASET_DIR and optional RADIOLOGY_UI_TOKEN
docker compose up -d --build
```

### 8.3 udocker Runtime Path (Fallback)

Use `udocker` when native Docker/Podman is unavailable. The runtime contract is the same image and environment variables as native OCI runtime.

```bash
# Example pattern (image must exist locally or be pulled beforehand)
python udocker.py run \
  -p 8000:8000 \
  -v /path/to/data/dataset:/data:rw \
  -e RADIOLOGY_UI_TOKEN=mysecret \
  -e ALLOW_DATA_MUTATIONS=true \
  radiology-ui:1.0
```

### 8.4 Configuration (Environment Variables)

| Variable                    | Default          | Description                                                  |
|----------------------------|------------------|--------------------------------------------------------------|
| `RADIOLOGY_UI_TOKEN`       | `""` (no auth)   | Bearer token for API protection (`/api/*`, except health)    |
| `DATA_ROOT`                | `/data`          | Mount point for dataset files                                |
| `LOG_LEVEL`                | `info`           | Python logging level                                         |
| `ALLOW_DATA_MUTATIONS`     | `false`          | Enables review apply (reclassify/delete) filesystem writes   |
| `PORT`                     | `8000`           | Uvicorn bind port                                            |
| `STATIC_ROOT`              | `/app/static`    | Frontend static build directory served by backend            |
| `VITE_AUTH_TOKEN_STORAGE`  | `memory`         | Frontend token persistence mode (`memory` or `local`)        |

### 8.5 Compose Host Variables (`.env`)

| Variable       | Default            | Description                                  |
|----------------|--------------------|----------------------------------------------|
| `DATASET_DIR`  | `./data/dataset`   | Host dataset directory mounted at `/data` (rw for review apply) |
| `WEBUI_PORT`   | `8000`             | Host port mapped to container `8000`         |

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
| 8  | If no mask exists, 3D panel shows an empty-state view with "No segmentation available" messaging.       | Manual test        |
| 9  | Double-click on any panel expands it; double-click again restores 2×2.                                 | Manual test        |
| 10 | VOI `.npy` series are loadable and displayable with the same interaction as NIfTI series.              | Manual test        |
| 11 | Application runs via `docker compose up` (native Docker/Podman) or equivalent `udocker run` fallback. | Deployment test    |
| 12 | Dataset mutations occur only through `review/apply` when `ALLOW_DATA_MUTATIONS=true`; no hard deletes. | Audit / strace     |
| 13 | Setting `RADIOLOGY_UI_TOKEN` blocks unauthenticated `/api/*` access (except `/api/health`).           | Manual test        |
| 14 | Last-viewed patient and W/L settings persist across browser refresh.                                   | Manual test        |
| 15 | Viewer group filter + patient dropdown + next button navigate patients correctly in filtered order.    | Manual test        |
| 16 | Applying queued `NC/ART/VEN` reclassify updates NIfTI manifest phase/protocol_source when available.   | Manual test        |
| 17 | Applying queued delete moves NIfTI/SEG and VOI image/mask files into recycle paths.                    | Manual test        |
| 18 | `decisions.json`, `reclassification_log.json`, and `deletion_log.json` are appended per apply batch.   | Manual test        |
| 19 | With `ALLOW_DATA_MUTATIONS=false`, review apply endpoint returns `409` and no dataset files are changed.| Manual test        |

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

*End of SRS v1.2*
