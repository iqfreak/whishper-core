"""voicelang — desktop app entry point (also the PyInstaller entry).

``voicelang.exe`` (or ``uv run python voicelang_app.py``) opens the settings
GUI. The GUI starts the live pipeline as a child process: in the packaged
app that child is this same exe with ``--run`` (a frozen app cannot do
``python -m voicelang_core.run``); in the source layout it is
``python -m voicelang_core.run --config <path>``.

CLI parity (spec §8): any headless flag (--headless / --list-devices /
--list-engines / --config / --source / ...) routes to the CLI even in the
packaged exe — the GUI only opens when NO CLI flags are given.

In the packaged (--windowed) GUI build stdout/stderr are None; the ``--run``
child re-points them at %LOCALAPPDATA%/voicelang/run.log so console caption
output and errors survive for debugging.
"""
import os
import sys

# Flags that mean "headless CLI, not the GUI" (spec §8 parity).
_CLI_FLAGS = {
    "--headless", "--list-devices", "--list-engines", "--config",
    "--source", "--source-device", "--source-app", "--input", "--display",
    "--transcriber", "--language", "--target-language", "--translate",
    "--translate-url", "--translate-key", "--model-arch",
}


def _wants_cli(argv: list[str]) -> bool:
    """True when argv asks for the headless CLI instead of the settings GUI."""
    if "--run" in argv:
        return True
    if "--headless" in argv:
        return True
    return any(a.split("=", 1)[0] in _CLI_FLAGS for a in argv[1:])


def _redirect_frozen_io() -> None:
    if not getattr(sys, "frozen", False) or sys.stdout is not None:
        return
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    log_dir = os.path.join(root, "voicelang")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")
    sys.stdout = sys.stderr = open(log_path, "a", encoding="utf-8")  # noqa: SIM115


def main() -> None:
    _redirect_frozen_io()
    if _wants_cli(sys.argv):
        sys.argv = [a for a in sys.argv if a not in ("--run", "--headless")]
        from voicelang_core.run import main as run_main

        run_main()
        return
    from voicelang_core.gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()