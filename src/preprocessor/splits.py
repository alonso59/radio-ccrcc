
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split

RARE = "__rare__"


def _safe_stratify(labels: List[str], min_count: int) -> List[str]:
    counts = Counter(labels)
    return [c if counts[c] >= min_count else RARE for c in labels]


def _gather_files(ids: List[str], lookup: Dict[str, List[str]]) -> List[str]:
    return [fp for pid in ids for fp in lookup[pid]]


def _pick_by_mask(items, mask, value, *, invert=False):
    if invert:
        return [x for x, m in zip(items, mask) if m != value]
    return [x for x, m in zip(items, mask) if m == value]


def scan_patients(images_dir: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Not found: {images_dir}")

    files: Dict[str, List[str]] = {}
    classes: Dict[str, str] = {}

    for path in sorted(images_dir.rglob("*.npy")):
        parts = path.parent.relative_to(images_dir).parts  # (class, patient, phase)
        if len(parts) < 2:
            continue
        cls, pid = parts[0], parts[1]
        files.setdefault(pid, []).append(str(path.resolve()))
        classes[pid] = cls

    return files, classes


def build_splits(
    files: Dict[str, List[str]],
    classes: Dict[str, str],
    *,
    n_folds: int = 5,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    patients = list(files)
    labels = [classes[p] for p in patients]

    # ── hold-out test set ────────────────────────────────────────────────
    strat = _safe_stratify(labels, min_count=2)
    rare_count = strat.count(RARE)

    if rare_count == 0 or rare_count >= 2:
        train_ids, test_ids = train_test_split(
            patients, test_size=test_ratio, stratify=strat, random_state=seed,
        )
    else:
        # Can't stratify a single-member rare bucket — force it into train.
        forced   = _pick_by_mask(patients, strat, RARE)
        elig_ids = _pick_by_mask(patients, strat, RARE, invert=True)
        elig_lbl = _pick_by_mask(strat, strat, RARE, invert=True)
        rest, test_ids = train_test_split(
            elig_ids, test_size=test_ratio, stratify=elig_lbl, random_state=seed,
        )
        train_ids = forced + rest

    # ── K-fold cross-validation on train set ─────────────────────────────
    fold_labels = _safe_stratify([classes[p] for p in train_ids], min_count=n_folds)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (ti, vi) in enumerate(skf.split(train_ids, fold_labels)):
        t_ids = [train_ids[i] for i in ti]
        v_ids = [train_ids[i] for i in vi]
        folds.append({
            "fold": fold_idx,
            "train_case_ids": t_ids,
            "val_case_ids": v_ids,
            "train_files": _gather_files(t_ids, files),
            "val_files": _gather_files(v_ids, files),
        })

    return {
        "params": {"n_folds": n_folds, "test_ratio": test_ratio, "seed": seed},
        "folds": folds,
        "test_case_ids": test_ids,
        "test_files": _gather_files(test_ids, files),
    }


# ── public entry point ───────────────────────────────────────────────────

def run_splits(cfg: Dict) -> None:
    output_dir = Path(cfg["OUTPUT_DIR"]).resolve()
    images_dir = output_dir / cfg.get("VOI_SUBFOLDER", "images")

    files, classes = scan_patients(images_dir)
    if not files:
        print("No patients found — run preprocessing first.")
        return

    n_folds = cfg.get("N_FOLDS", 5)
    dist = dict(sorted(Counter(classes.values()).items()))
    print(f"Found {len(files)} patients | Classes: {dist}")

    for cls, n in dist.items():
        if n < 2:
            print(f"  ! '{cls}' ({n} patient) — forced into train only")
        elif n < n_folds:
            print(f"  ! '{cls}' ({n} patients) — pooled as rare for fold stratification")

    splits = build_splits(
        files, classes,
        n_folds=n_folds,
        test_ratio=cfg.get("TEST_RATIO", 0.1),
        seed=cfg.get("RANDOM_SEED", 42),
    )

    out_path = output_dir / "splits.json"
    out_path.write_text(json.dumps(splits, indent=2))
    print(f"Saved -> {out_path}")
