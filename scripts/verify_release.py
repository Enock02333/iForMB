from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
FORBIDDEN = {".pt", ".pth", ".pkl", ".joblib", ".log"}
violations: list[str] = []
windows_home = "C:" + chr(92) + "Users" + chr(92)
mac_home = "/" + "Users" + "/"
linux_home = "/" + "home" + "/"

for path in ROOT.rglob("*"):
    if not path.is_file() or any(part in SKIP for part in path.parts):
        continue
    relative = path.relative_to(ROOT)
    if path.suffix.lower() in FORBIDDEN:
        violations.append(f"forbidden artifact: {relative}")
    if path.stat().st_size > 10 * 1024 * 1024:
        violations.append(f"file exceeds 10 MB: {relative}")
    if path.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cff"} or path.name in {"LICENSE", ".gitignore"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if windows_home in text:
            violations.append(f"absolute Windows user path: {relative}")
        if mac_home in text:
            violations.append(f"absolute macOS user path: {relative}")
        if re.search(re.escape(linux_home) + r"[A-Za-z0-9_.-]+/", text):
            violations.append(f"absolute Linux user path: {relative}")

if violations:
    print("\n".join(sorted(set(violations))))
    sys.exit(1)
print("Release verification passed.")
