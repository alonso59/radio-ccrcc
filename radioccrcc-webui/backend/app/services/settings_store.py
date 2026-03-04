from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.models.settings import DatasetViewerSettings


class SettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or self._default_path()

    def load(self) -> dict[str, DatasetViewerSettings]:
        if not self.path.exists():
            return {}

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Settings file '{self.path}' is not valid JSON") from exc
        except OSError as exc:
            raise RuntimeError(f"Unable to read settings file '{self.path}'") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"Settings file '{self.path}' must contain a JSON object")

        return {
            dataset_id: DatasetViewerSettings.model_validate(value)
            for dataset_id, value in payload.items()
        }

    def save(
        self,
        settings: dict[str, DatasetViewerSettings],
    ) -> dict[str, DatasetViewerSettings]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            dataset_id: config.model_dump(mode="json")
            for dataset_id, config in settings.items()
        }

        try:
            with NamedTemporaryFile(
                "w",
                dir=self.path.parent,
                prefix=self.path.stem + ".",
                suffix=".tmp",
                encoding="utf-8",
                delete=False,
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                temp_path = Path(handle.name)
            temp_path.replace(self.path)
        except OSError as exc:
            raise RuntimeError(f"Unable to write settings file '{self.path}'") from exc

        return settings

    @staticmethod
    def _default_path() -> Path:
        home_path = Path.home() / ".radiology-webui" / "settings.json"
        if SettingsStore._is_writable(home_path.parent):
            return home_path
        return Path("/tmp/radiology-webui/settings.json")

    @staticmethod
    def _is_writable(directory: Path) -> bool:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write-test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False


settings_store = SettingsStore()
