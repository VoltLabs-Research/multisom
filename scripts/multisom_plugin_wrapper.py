"""VOLT entrypoint for the MultiSOM plugin.

The host either imports this as a persistent stub and calls
``process(frame, config)``, or runs it as a CLI with the same argument list.
Both forward to ``volt.run``. STDOUT is reserved for the host IPC protocol;
logs go to STDERR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = Path(os.environ.get("PLUGIN_PROJECT_DIR", SCRIPT_DIR.parent)).resolve()
for _path in (PLUGIN_ROOT, SCRIPT_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

if __name__ == "__main__" and "--describe" in sys.argv[1:]:
    import importlib.util
    import json

    _spec = importlib.util.spec_from_file_location(
        "multisom_interface", PLUGIN_ROOT / "volt" / "interface.py"
    )
    _interface = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_interface)
    print(json.dumps(_interface.DESCRIPTOR))
    raise SystemExit(0)

from volt import run  # noqa: E402


def process(frame, config):
    """Volt persistent-stub entrypoint. The frame is ignored — features come
    from the upstream parquets in the frame's output directory."""
    del frame
    return run(config["args"])


if __name__ == "__main__":
    try:
        run(sys.argv[1:])
    except ValueError as error:
        sys.exit(f"multisom: {error}")
