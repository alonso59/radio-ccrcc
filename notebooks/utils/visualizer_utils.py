"""
Shared utilities for medical image visualizers (nifti, voi, etc.)

Provides common functions for:
  • HU value normalization
  • Segmentation/mask overlay rendering
  • Decisions persistence (JSON I/O)
  • Orthogonal view rendering (axial, coronal, sagittal)
  • Widget callbacks and helpers
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Any
from skimage import measure
import ipywidgets as widgets
from IPython.display import clear_output


# ── Crosshair Colors ──────────────────────────────────────────────────────────
COLOR_AXIAL    = 'yellow'   # shown in coronal & sagittal
COLOR_CORONAL  = 'red'      # shown in axial & sagittal
COLOR_SAGITTAL = 'green'    # shown in axial & coronal


# ── HU Normalization ──────────────────────────────────────────────────────────

def hu_to_display(volume: np.ndarray, hu_min: float = -150, hu_max: float = 250) -> np.ndarray:
    """
    Clip a HU volume to [hu_min, hu_max] and normalise to [0, 1].
    
    Args:
        volume: Input array (any shape)
        hu_min: Lower HU clipping value (default: -150)
        hu_max: Upper HU clipping value (default: 250)
    
    Returns:
        Normalized array in [0, 1]
    """
    clipped = np.clip(volume.astype(np.float32), hu_min, hu_max)
    return (clipped - hu_min) / (hu_max - hu_min)


# ── Segmentation/Mask Overlay ──────────────────────────────────────────────────

# Default layer configuration for multi-class segmentation
DEFAULT_LAYER_CONFIG = {
    1: {'name': 'Kidney', 'color': 'cyan',    'alpha': 0.15, 'linewidth': 1.5},
    2: {'name': 'Tumor',  'color': 'yellow',  'alpha': 0.20, 'linewidth': 2.0},
    3: {'name': 'Cyst',   'color': 'magenta', 'alpha': 0.15, 'linewidth': 1.5},
}


def overlay_binary_mask(
    ax_obj,
    mask_slice_2d: np.ndarray | None,
    label: int = 2,
    alpha: float = 0.08,
    color: str = 'yellow',
    linewidth: float = 1.0
):
    """
    Draw a filled overlay + contour border for a binary mask on an axis.
    
    Args:
        ax_obj: matplotlib Axes object to draw on
        mask_slice_2d: 2D slice of mask array (already transposed to match image)
        label: Voxel value treated as "positive" in the mask
        alpha: Fill transparency [0, 1]
        color: Color of overlay and contour
        linewidth: Contour border linewidth
    """
    if mask_slice_2d is None:
        return
    
    binary = (mask_slice_2d == label).astype(np.uint8)
    
    if binary.max() == 0:
        return  # no mask in this slice

    # ── Filled overlay ────────────────────────────────────────────────────────
    rgba = np.zeros((*binary.shape, 4), dtype=np.float32)
    rgb = mcolors.to_rgb(color)
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = binary * alpha
    ax_obj.imshow(rgba, origin='lower', aspect='auto', interpolation='nearest')

    # ── Contour border ────────────────────────────────────────────────────────
    contours = measure.find_contours(binary, level=0.5)
    for contour in contours:
        ax_obj.plot(contour[:, 1], contour[:, 0],
                    color=color, linewidth=linewidth)


def overlay_multi_layer_mask(
    ax_obj,
    mask_slice_2d: np.ndarray | None,
    visible_labels: dict[int, bool],
    layer_config: dict[int, dict[str, Any]] | None = None
):
    """
    Render multiple anatomical structures with independent colors and alphas.
    
    Args:
        ax_obj: matplotlib Axes object to draw on
        mask_slice_2d: 2D slice with integer labels (0=bg, 1=kidney, 2=tumor, 3=cyst)
        visible_labels: {1: True, 2: False, 3: True} — which layers to show
        layer_config: Dict with 'color', 'alpha', 'linewidth' per label
    """
    if mask_slice_2d is None:
        return
    
    if layer_config is None:
        layer_config = DEFAULT_LAYER_CONFIG
    
    for label, is_visible in sorted(visible_labels.items()):
        if not is_visible or label not in layer_config:
            continue
        
        config = layer_config[label]
        binary = (mask_slice_2d == label).astype(np.uint8)
        
        if binary.max() == 0:
            continue  # No pixels for this label in this slice
        
        # ── Filled overlay ────────────────────────────────────────────────────
        rgba = np.zeros((*binary.shape, 4), dtype=np.float32)
        rgb = mcolors.to_rgb(config['color'])
        rgba[..., 0] = rgb[0]
        rgba[..., 1] = rgb[1]
        rgba[..., 2] = rgb[2]
        rgba[..., 3] = binary * config['alpha']
        ax_obj.imshow(rgba, origin='lower', aspect='auto', interpolation='nearest')
        
        # ── Contour border ────────────────────────────────────────────────────
        contours = measure.find_contours(binary, level=0.5)
        for contour in contours:
            ax_obj.plot(contour[:, 1], contour[:, 0],
                       color=config['color'], linewidth=config['linewidth'])


# ── Decisions Persistence ─────────────────────────────────────────────────────

def load_decisions(decisions_file: Path) -> dict:
    """
    Load decisions from a JSON file.
    
    Args:
        decisions_file: Path to decisions.json
    
    Returns:
        Dict {key: entry_dict}, or {} if file doesn't exist
    """
    if decisions_file.exists():
        with open(decisions_file, 'r') as f:
            entries = json.load(f)
        return {e['key']: e for e in entries}
    return {}


def save_decisions(decisions: dict, decisions_file: Path):
    """
    Persist decisions dict to a JSON file.
    
    Args:
        decisions: Dict {key: entry_dict}
        decisions_file: Path to write decisions.json
    """
    decisions_file.parent.mkdir(parents=True, exist_ok=True)
    with open(decisions_file, 'w') as f:
        json.dump(list(decisions.values()), f, indent=2)


# ── Orthogonal View Rendering ─────────────────────────────────────────────────

def render_orthogonal_views(
    state,
    output_widget: widgets.Output,
    overlay_func=None,
    case_title_suffix: str = "",
    visible_labels: dict[int, bool] | None = None
):
    """
    Render axial | coronal | sagittal views into an output widget.
    
    Args:
        state: CaseReviewState object with image, disp, seg/mask, and slice indices
        output_widget: ipywidgets.Output to render into
        overlay_func: Callable(ax, slice_2d) or Callable(ax, slice_2d, visible_labels) for overlay
        case_title_suffix: Extra info to append to the title
        visible_labels: Optional dict {label: bool} to pass to overlay_func (for multi-layer)
    
    Expects state to have:
        .image, .disp (HU-normalized), .seg/.mask, .ax_idx, .cor_idx, .sag_idx
        .shape (property returning (X, Y, Z))
        .current_case() -> dict with 'key' field
        .get_decision() -> str | None
    """
    with output_widget:
        clear_output(wait=True)

        if state.image is None:
            print("No image loaded.")
            return

        az  = state.ax_idx
        cy  = state.cor_idx
        sx  = state.sag_idx
        X, Y, Z = state.shape

        # ── Extract and transpose slices ──────────────────────────────────────
        ax_img  = state.disp[:, :, az].T    # (Y, X)
        cor_img = state.disp[:, cy, :].T    # (Z, X)
        sag_img = state.disp[sx, :, :].T    # (Z, Y)
        
        # Get overlay slices (if available)
        overlay_attr = None
        if hasattr(state, 'seg') and state.seg is not None:
            overlay_attr = 'seg'
        elif hasattr(state, 'mask') and state.mask is not None:
            overlay_attr = 'mask'
        
        ax_overlay  = None
        cor_overlay = None
        sag_overlay = None
        
        if overlay_attr is not None:
            overlay_data = getattr(state, overlay_attr)
            ax_overlay  = overlay_data[:, :, az].T
            cor_overlay = overlay_data[:, cy, :].T
            sag_overlay = overlay_data[sx, :, :].T

        # ── Figure ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                                 facecolor='black',
                                 gridspec_kw={'wspace': 0.04})

        titles    = ['Axial',    'Coronal',   'Sagittal']
        imgs      = [ax_img,     cor_img,     sag_img]
        overlays  = [ax_overlay, cor_overlay, sag_overlay]
        # (hline_val, hline_color, vline_val, vline_color)
        lines = [
            (cy,  COLOR_CORONAL,  sx,  COLOR_SAGITTAL),   # Axial
            (az,  COLOR_AXIAL,    sx,  COLOR_SAGITTAL),   # Coronal
            (az,  COLOR_AXIAL,    cy,  COLOR_CORONAL),    # Sagittal
        ]

        for a, title, img, overlay_slice, (hv, hc, vv, vc) in zip(
                axes, titles, imgs, overlays, lines):

            a.imshow(img, cmap='gray', origin='lower', aspect='auto',
                     vmin=0, vmax=1, interpolation='bilinear')
            
            # Draw overlay if function provided
            if overlay_func is not None and overlay_slice is not None:
                # Check if overlay function accepts visible_labels parameter
                import inspect
                sig = inspect.signature(overlay_func)
                if 'visible_labels' in sig.parameters and visible_labels is not None:
                    overlay_func(a, overlay_slice, visible_labels)
                else:
                    overlay_func(a, overlay_slice)

            # Crosshair lines
            a.axhline(y=hv, color=hc, linewidth=1.2, alpha=0.85)
            a.axvline(x=vv, color=vc, linewidth=1.2, alpha=0.85)

            a.set_title(title, color='white', fontsize=12, pad=4)
            a.axis('off')

        # ── Case info in suptitle ─────────────────────────────────────────────
        case = state.current_case()
        decision = state.get_decision()
        dec_str  = f"  [{decision}]" if decision else "  [undecided]"
        fig.suptitle(
            f"Case {state.idx + 1}/{state.n_cases}  ·  {case['key']}{dec_str}{case_title_suffix}\n"
            f"ax={az}/{Z-1}  cor={cy}/{Y-1}  sag={sx}/{X-1}",
            color='white', fontsize=10, y=1.01
        )
        fig.patch.set_facecolor('black')

        plt.show()


# ── Widget Helpers ────────────────────────────────────────────────────────────

def decision_badge(decision: str | None) -> str:
    """Format decision status as HTML badge."""
    if decision is None:
        return "<span style='color:#aaa;'>[undecided]</span>"
    if decision == 'delete':
        return "<span style='color:#ff4444;font-weight:bold;'>⚠ MARKED FOR DELETION</span>"
    return f"<span style='color:#44cc44;font-weight:bold;'>✓ {decision.upper()}</span>"


def sync_sliders(state, sl_ax: widgets.IntSlider, sl_cor: widgets.IntSlider, 
                 sl_sag: widgets.IntSlider, updating_flag: list):
    """
    Push current state slice indices into sliders without triggering callbacks.
    
    Args:
        state: CaseReviewState object
        sl_ax, sl_cor, sl_sag: IntSlider widgets
        updating_flag: Single-element list [bool] to set/unset during sync
    """
    updating_flag[0] = True
    sl_ax.max  = state.max_ax()  ; sl_ax.value  = state.ax_idx
    sl_cor.max = state.max_cor() ; sl_cor.value = state.cor_idx
    sl_sag.max = state.max_sag() ; sl_sag.value = state.sag_idx
    updating_flag[0] = False


def sync_case_index(state, txt_case_idx: widgets.BoundedIntText, updating_flag: list):
    """
    Update the case index text box without triggering callbacks.
    
    Args:
        state: CaseReviewState object
        txt_case_idx: BoundedIntText widget
        updating_flag: Single-element list [bool] to set/unset during sync
    """
    updating_flag[0] = True
    txt_case_idx.value = state.idx + 1  # Display 1-indexed
    updating_flag[0] = False


def create_layer_checkboxes(
    layer_config: dict[int, dict[str, Any]] | None = None,
    default_visible: dict[int, bool] | None = None,
) -> dict[int, widgets.Checkbox]:
    """
    Create checkbox widgets for layer visibility control.
    
    Args:
        layer_config: Dict with layer metadata (name, color, etc.)
        default_visible: Dict {label: bool} for initial checkbox states
    
    Returns:
        Dict {label: Checkbox widget}
    """
    if layer_config is None:
        layer_config = DEFAULT_LAYER_CONFIG
    
    if default_visible is None:
        default_visible = {label: True for label in layer_config.keys()}
    
    checkboxes = {}
    for label in sorted(layer_config.keys()):
        config = layer_config[label]
        checkboxes[label] = widgets.Checkbox(
            value=default_visible.get(label, True),
            description=config['name'],
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='100px')
        )
    return checkboxes


def create_overlay_control_panel(
    checkboxes: dict[int, widgets.Checkbox],
    layer_config: dict[int, dict[str, Any]] | None = None,
) -> widgets.HBox:
    """
    Create a styled control panel for overlay checkboxes.
    
    Args:
        checkboxes: Dict {label: Checkbox widget}
        layer_config: Dict with layer metadata for color indicators
    
    Returns:
        HBox widget with labeled checkboxes
    """
    if layer_config is None:
        layer_config = DEFAULT_LAYER_CONFIG
    
    # Create label with color indicators
    label_parts = []
    for label in sorted(checkboxes.keys()):
        if label in layer_config:
            config = layer_config[label]
            # Use colored squares as visual indicators
            label_parts.append(f"<span style='color:{config['color']};'>■</span> {config['name']}")
    
    title = widgets.HTML(
        value=f"<b style='color:#ddd;'>Overlay:</b> {' │ '.join(label_parts)}",
        layout=widgets.Layout(width='auto', margin='0 10px 0 0')
    )
    
    checkbox_widgets = [checkboxes[label] for label in sorted(checkboxes.keys())]
    
    return widgets.HBox(
        [title] + checkbox_widgets,
        layout=widgets.Layout(
            justify_content='center',
            align_items='center',
            padding='6px 0',
            gap='8px'
        )
    )


def get_visible_labels(layer_checkboxes: dict) -> dict[int, bool]:
    """Return visibility map from a label->Checkbox dictionary."""
    return {label: checkbox.value for label, checkbox in layer_checkboxes.items()}


def build_layer_status_suffix(
    visible_labels: dict[int, bool],
    layer_config: dict[int, dict[str, Any]] | None = None,
    has_overlay: bool = True,
) -> str:
    """
    Build compact status suffix like: "  [K:✓ T:✓ C:✗]".
    """
    if not has_overlay:
        return ""

    if layer_config is None:
        layer_config = DEFAULT_LAYER_CONFIG

    layer_status = []
    for label in sorted(visible_labels.keys()):
        if label in layer_config:
            symbol = '✓' if visible_labels[label] else '✗'
            name_short = layer_config[label]['name'][0]
            layer_status.append(f"{name_short}:{symbol}")
    return f"  [{' '.join(layer_status)}]"


def create_viewer_layout(
    status_html: widgets.HTML,
    out: widgets.Output,
    overlay_panel: widgets.HBox,
    sliders: widgets.VBox,
    nav_row: widgets.HBox,
    btn_row: widgets.HBox,
) -> widgets.VBox:
    """Create the standard visualizer UI container used by both notebooks."""
    return widgets.VBox(
        [status_html, out, overlay_panel, sliders, nav_row, btn_row],
        layout=widgets.Layout(
            border='1px solid #333',
            border_radius='8px',
            padding='8px',
            background_color='#121212',
        )
    )


# ── Callback Builders ──────────────────────────────────────────────────────────

def make_slider_callback(state, output_widget: widgets.Output, axis: str):
    """
    Create a slider change callback for axial, coronal, or sagittal.
    
    Args:
        state: CaseReviewState object
        output_widget: ipywidgets.Output widget to render into
        axis: 'ax', 'cor', or 'sag'
        
    Returns:
        Callback function (change) -> None
    """
    def callback(change):
        state.__dict__[f'{axis}_idx'] = change['new']
        from IPython.display import display
        render_orthogonal_views(state, output_widget)
    return callback


def make_classification_callback(state, output_widget: widgets.Output, 
                                  sliders_sync_func, case_sync_func, status_update_func,
                                  updating_flag: list):
    """
    Create a classification button callback.
    
    Args:
        state: CaseReviewState object
        output_widget: ipywidgets.Output widget
        sliders_sync_func: Function to sync sliders
        case_sync_func: Function to sync case index
        status_update_func: Function to update status HTML
        updating_flag: Single-element list [bool]
        
    Returns:
        Callback factory: (protocol_str) -> (button_click) -> None
    """
    def make_handler(protocol: str):
        def handler(_):
            state.set_decision(protocol)
            state.go_next()
            sliders_sync_func()
            case_sync_func()
            status_update_func()
            render_orthogonal_views(state, output_widget)
        return handler
    return make_handler
