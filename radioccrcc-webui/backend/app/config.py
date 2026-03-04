from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    data_root: str
    radiology_ui_token: str
    log_level: str
    port: int


def _parse_port(value: str | None) -> int:
    if value is None:
        return 8000

    try:
        return int(value)
    except (TypeError, ValueError):
        return 8000


def get_settings() -> Settings:
    return Settings(
        data_root=os.environ.get("DATA_ROOT", "../../data/dataset"),
        radiology_ui_token=os.environ.get("RADIOLOGY_UI_TOKEN", ""),
        log_level=os.environ.get("LOG_LEVEL", "info"),
        port=_parse_port(os.environ.get("PORT")),
    )
