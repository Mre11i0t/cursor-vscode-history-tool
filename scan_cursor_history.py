#!/usr/bin/env python3
"""
Scan Cursor/VS Code local history to list deleted and modified files under a workspace,
show diffs, and optionally restore deleted files from their latest snapshots.

This version adds:
  - Colored output with Rich (auto-fallback to plain text)
  - Progress bar with tqdm/Rich (auto-fallback to simple counters)
  - Glob filters (--include/--exclude)
  - Time window filters (--since)
  - Multiple export formats (human/json/csv/ndjson)
  - Safer restore workflow with dry-run and overwrite control
  - Cross-platform default path resolution for Cursor/VS Code

Author: <your-name>
License: MIT
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses as dc
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, unquote
import csv

__version__ = "1.3.0"

# --------------------------- Optional, pretty output ---------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None

try:
    from tqdm import tqdm  # type: ignore
    TQDM = True
except Exception:
    TQDM = False

# --------------------------- Defaults & helpers ---------------------------

def default_history_paths() -> List[Path]:
    paths: List[Path] = []
    home = Path.home()
    # macOS Cursor/VS Code
    paths.append(home / "Library/Application Support/Cursor/User/History")
    paths.append(home / "Library/Application Support/Code/User/History")
    # Windows Cursor/VS Code
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "Cursor/User/History")
        paths.append(Path(appdata) / "Code/User/History")
    # Linux VS Code
    paths.append(home / ".config/Code/User/History")
    # De-dup while preserving order
    dedup: List[Path] = []
    seen = set()
    for p in paths:
        if str(p) not in seen:
            seen.add(str(p))
            dedup.append(p)
    return dedup

DEFAULT_HISTORY_DIR = next((p for p in default_history_paths() if p.exists()), default_history_paths()[0])
DEFAULT_WORKSPACE_DIR = str(Path.cwd())

@dc.dataclass
class HistoryEntry:
    id: Optional[str]
    source: Optional[str]
    timestamp_ms: Optional[int]

@dc.dataclass
class HistoryRecord:
    history_dir: Path
    resource_path: Path
    entries: List[HistoryEntry]

    @property
    def last_timestamp_ms(self) -> int:
        ts = [e.timestamp_ms for e in self.entries if e.timestamp_ms is not None]
        return max(ts) if ts else 0

    @property
    def sources(self) -> List[str]:
        return [e.source for e in self.entries if e.source]

    def latest_existing_snapshot(self) -> Optional[Tuple[int, Path]]:
        existing: List[Tuple[int, Path]] = []
        for e in self.entries:
            if not e.id or e.timestamp_ms is None:
                continue
            candidate = self.history_dir / e.id
            if candidate.exists() and candidate.is_file():
                existing.append((e.timestamp_ms, candidate))
        if not existing:
            return None
        existing.sort(key=lambda t: t[0], reverse=True)
        return existing[0]

    def latest_existing_snapshot_file(self) -> Optional[Path]:
        latest = self.latest_existing_snapshot()
        return latest[1] if latest else None

# --------------------------- Parsing & IO ---------------------------

def parse_entries_json(entries_path: Path) -> Optional[HistoryRecord]:
    try:
        raw = entries_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return None

    resource_uri = data.get("resource")
    if not isinstance(resource_uri, str):
        return None

    resource_path = file_uri_to_path(resource_uri)
    if resource_path is None:
        return None

    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        raw_entries = []

    entries: List[HistoryEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        entries.append(
            HistoryEntry(
                id=item.get("id"),
                source=item.get("source"),
                timestamp_ms=item.get("timestamp"),
            )
        )

    return HistoryRecord(
        history_dir=entries_path.parent,
        resource_path=resource_path,
        entries=entries,
    )

def file_uri_to_path(resource_uri: str) -> Optional[Path]:
    try:
        parsed = urlparse(resource_uri)
        if parsed.scheme != "file":
            return None
        path = unquote(parsed.path)
        return Path(path)
    except Exception:
        return None

def read_bytes_safely(p: Path) -> Optional[bytes]:
    try:
        return p.read_bytes()
    except Exception:
        return None

def sha256_digest(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def unified_diff_text(a_bytes: bytes, b_bytes: bytes, a_label: str, b_label: str) -> str:
    try:
        a_text = a_bytes.decode("utf-8", errors="replace").splitlines()
        b_text = b_bytes.decode("utf-8", errors="replace").splitlines()
        diff_lines = difflib.unified_diff(a_text, b_text, fromfile=a_label, tofile=b_label, lineterm="")
        return "\n".join(diff_lines)
    except Exception:
        return ""

# --------------------------- Model ---------------------------

@dc.dataclass
class FileStatus:
    resource_path: Path
    history_dir: Path
    last_edit_at: Optional[dt.datetime]
    status: str  # "DELETED" | "MODIFIED" | "UNCHANGED"
    latest_snapshot: Optional[Path]
    latest_snapshot_ts_ms: Optional[int] = None
    reason: Optional[str] = None

# --------------------------- Filters ---------------------------

def matches_globs(path: Path, patterns: List[str]) -> bool:
    if not patterns:
        return True
    for pat in patterns:
        if path.match(pat):
            return True
    return False

def parse_since(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    s = s.strip()
    # absolute date YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return dt.datetime(y, mo, d, tzinfo=dt.timezone.utc)
    # relative like 3d, 12h, 30m
    m = re.fullmatch(r"(\d+)([dhm])", s)
    if m:
        n, unit = m.groups()
        n = int(n)
        now = dt.datetime.now(tz=dt.timezone.utc)
        if unit == "d":
            return now - dt.timedelta(days=n)
        if unit == "h":
            return now - dt.timedelta(hours=n)
        if unit == "m":
            return now - dt.timedelta(minutes=n)
    raise ValueError(f"Invalid --since value: {s}")

# --------------------------- Core processing ---------------------------

def process_history_dir(entries_json_path: Path, workspace_dir: Path) -> Optional[FileStatus]:
    record = parse_entries_json(entries_json_path)
    if record is None:
        return None

    try:
        resource_path = record.resource_path.resolve()
    except Exception:
        resource_path = record.resource_path

    try:
        workspace_dir_resolved = workspace_dir.resolve()
    except Exception:
        workspace_dir_resolved = workspace_dir

    try:
        if workspace_dir_resolved not in resource_path.parents and resource_path != workspace_dir_resolved:
            return None
    except Exception:
        if not str(resource_path).startswith(str(workspace_dir)):
            return None

    last_ts_ms = record.last_timestamp_ms
    last_edit_at = dt.datetime.fromtimestamp(last_ts_ms / 1000.0, tz=dt.timezone.utc) if last_ts_ms else None

    exists_now = resource_path.exists()
    latest_snapshot = record.latest_existing_snapshot()
    latest_snapshot_ts = latest_snapshot[0] if latest_snapshot else None
    latest_snapshot_path = latest_snapshot[1] if latest_snapshot else None

    if not exists_now:
        return FileStatus(
            resource_path=resource_path,
            history_dir=record.history_dir,
            last_edit_at=last_edit_at,
            status="DELETED",
            latest_snapshot=latest_snapshot_path,
            latest_snapshot_ts_ms=latest_snapshot_ts,
            reason="File missing on disk",
        )

    if latest_snapshot_path is None:
        status = "UNCHANGED" if not record.entries else "MODIFIED"
        return FileStatus(
            resource_path=resource_path,
            history_dir=record.history_dir,
            last_edit_at=last_edit_at,
            status=status,
            latest_snapshot=None,
            latest_snapshot_ts_ms=None,
            reason="No snapshot present to compare",
        )

    current_bytes = read_bytes_safely(resource_path)
    snapshot_bytes = read_bytes_safely(latest_snapshot_path)
    if current_bytes is None or snapshot_bytes is None:
        return FileStatus(
            resource_path=resource_path,
            history_dir=record.history_dir,
            last_edit_at=last_edit_at,
            status="MODIFIED",
            latest_snapshot=latest_snapshot_path,
            latest_snapshot_ts_ms=latest_snapshot_ts,
            reason="Unable to read one or both files",
        )

    if sha256_digest(current_bytes) != sha256_digest(snapshot_bytes):
        return FileStatus(
            resource_path=resource_path,
            history_dir=record.history_dir,
            last_edit_at=last_edit_at,
            status="MODIFIED",
            latest_snapshot=latest_snapshot_path,
            latest_snapshot_ts_ms=latest_snapshot_ts,
            reason=None,
        )
    else:
        return FileStatus(
            resource_path=resource_path,
            history_dir=record.history_dir,
            last_edit_at=last_edit_at,
            status="UNCHANGED",
            latest_snapshot=latest_snapshot_path,
            latest_snapshot_ts_ms=latest_snapshot_ts,
            reason=None,
        )

# --------------------------- Output helpers ---------------------------

def print_human(statuses: List[FileStatus], include_diffs: bool) -> None:
    statuses_sorted = sorted(
        statuses,
        key=lambda s: (s.last_edit_at or dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)),
        reverse=True,
    )

    if RICH:
        table = Table(title=f"Cursor/VSCode History Scan — {len(statuses_sorted)} change(s)")
        table.add_column("Status")
        table.add_column("Path", overflow="fold")
        table.add_column("Last Edit (UTC)")
        table.add_column("History Store")
        for s in statuses_sorted:
            status = Text(s.status)
            if s.status == "DELETED":
                status.stylize("bold red")
            elif s.status == "MODIFIED":
                status.stylize("yellow")
            else:
                status.stylize("green")
            table.add_row(
                status,
                str(s.resource_path),
                s.last_edit_at.isoformat() if s.last_edit_at else "unknown",
                s.history_dir.name,
            )
        console.print(table)
    else:
        for s in statuses_sorted:
            ts_str = s.last_edit_at.isoformat() if s.last_edit_at else "unknown"
            print(f"{s.status}: {s.resource_path}  [last_edit={ts_str}]  (history={s.history_dir.name})")
            if s.reason:
                print(f"  reason: {s.reason}")

    if include_diffs:
        for s in statuses_sorted:
            if s.status == "MODIFIED" and s.latest_snapshot is not None:
                current_bytes = read_bytes_safely(s.resource_path) or b""
                snapshot_bytes = read_bytes_safely(s.latest_snapshot) or b""
                diff_txt = unified_diff_text(
                    snapshot_bytes,
                    current_bytes,
                    a_label=f"history:{s.latest_snapshot.name}",
                    b_label=f"current:{s.resource_path.name}",
                )
                if diff_txt:
                    if RICH:
                        console.rule(str(s.resource_path))
                        console.print(Panel.fit(Text(diff_txt)))
                    else:
                        print(f"\n--- Diff for {s.resource_path} ---\n{diff_txt}\n")


def print_json(statuses: List[FileStatus], include_diffs: bool) -> None:
    out = []
    for s in statuses:
        item: Dict[str, object] = {
            "status": s.status,
            "path": str(s.resource_path),
            "history_dir": str(s.history_dir),
            "last_edit": s.last_edit_at.isoformat() if s.last_edit_at else None,
            "latest_snapshot": str(s.latest_snapshot) if s.latest_snapshot else None,
            "latest_snapshot_ts": s.latest_snapshot_ts_ms,
            "reason": s.reason,
        }
        if include_diffs and s.status == "MODIFIED" and s.latest_snapshot is not None:
            current_bytes = read_bytes_safely(s.resource_path) or b""
            snapshot_bytes = read_bytes_safely(s.latest_snapshot) or b""
            item["diff"] = unified_diff_text(
                snapshot_bytes,
                current_bytes,
                a_label=f"history:{s.latest_snapshot.name}",
                b_label=f"current:{s.resource_path.name}",
            )
        out.append(item)
    print(json.dumps(out, indent=2))


def print_csv(statuses: List[FileStatus]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(["status", "path", "history_dir", "last_edit", "latest_snapshot", "latest_snapshot_ts", "reason"])
    for s in statuses:
        writer.writerow([
            s.status,
            str(s.resource_path),
            str(s.history_dir),
            s.last_edit_at.isoformat() if s.last_edit_at else "",
            str(s.latest_snapshot) if s.latest_snapshot else "",
            s.latest_snapshot_ts_ms or "",
            s.reason or "",
        ])


def print_ndjson(statuses: List[FileStatus]) -> None:
    for s in statuses:
        item: Dict[str, object] = {
            "status": s.status,
            "path": str(s.resource_path),
            "history_dir": str(s.history_dir),
            "last_edit": s.last_edit_at.isoformat() if s.last_edit_at else None,
            "latest_snapshot": str(s.latest_snapshot) if s.latest_snapshot else None,
            "latest_snapshot_ts": s.latest_snapshot_ts_ms,
            "reason": s.reason,
        }
        print(json.dumps(item))

# --------------------------- Restore ---------------------------

@dc.dataclass
class RestoreResult:
    path: Path
    restored: bool
    reason: Optional[str] = None

def restore_deleted_files(statuses: List[FileStatus], dry_run: bool = False, overwrite_if_exists: bool = False) -> List[RestoreResult]:
    results: List[RestoreResult] = []
    for s in statuses:
        if s.status != "DELETED":
            continue
        if s.latest_snapshot is None:
            results.append(RestoreResult(path=s.resource_path, restored=False, reason="No snapshot available"))
            continue
        dest = s.resource_path
        try:
            dest_parent = dest.parent
            if not dest_parent.exists() and not dry_run:
                dest_parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not overwrite_if_exists:
                results.append(RestoreResult(path=dest, restored=False, reason="File already exists"))
                continue
            if dry_run:
                results.append(RestoreResult(path=dest, restored=True, reason="dry-run"))
                continue
            data = read_bytes_safely(s.latest_snapshot)
            if data is None:
                results.append(RestoreResult(path=dest, restored=False, reason="Failed to read snapshot"))
                continue
            dest.write_bytes(data)
            ts_ms = s.latest_snapshot_ts_ms
            if ts_ms:
                ts = ts_ms / 1000.0
                os.utime(dest, (ts, ts))
            results.append(RestoreResult(path=dest, restored=True))
        except Exception as e:
            results.append(RestoreResult(path=dest, restored=False, reason=str(e)))
    return results

# --------------------------- Main ---------------------------

def iter_entries_json_paths(history_root: Path) -> Iterable[Path]:
    with os.scandir(history_root) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            entries_json = Path(entry.path) / "entries.json"
            if entries_json.exists() and entries_json.is_file():
                yield entries_json

def within_since(since: Optional[dt.datetime], last_edit_at: Optional[dt.datetime]) -> bool:
    if since is None:
        return True
    if last_edit_at is None:
        return False
    return last_edit_at >= since


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List deleted/modified files from Cursor/VSCode history within a workspace; show diffs; restore",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--history", default=str(DEFAULT_HISTORY_DIR), help="History root directory (per-file subfolders)")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE_DIR), help="Absolute path to the workspace to filter on")
    parser.add_argument("--output", choices=["human", "json", "csv", "ndjson"], default="human", help="Output format")
    parser.add_argument("--diff", action="store_true", help="Include unified diffs for modified files (slower)")
    parser.add_argument("--restore-deleted", action="store_true", help="Restore all deleted files from latest snapshot")
    parser.add_argument("--restore-dry-run", action="store_true", help="Do not write files; just report what would be restored")
    parser.add_argument("--restore-overwrite", action="store_true", help="Overwrite if target file already exists during restore")
    parser.add_argument("--max-workers", type=int, default=max(4, (os.cpu_count() or 4)), help="Parallel workers for scanning")
    parser.add_argument("--include", default="", help="Comma-separated glob(s) to include (e.g. **/*.ts,**/*.tsx)")
    parser.add_argument("--exclude", default="", help="Comma-separated glob(s) to exclude (e.g. **/node_modules/**)")
    parser.add_argument("--since", default="", help="Only show changes since date (YYYY-MM-DD) or relative window like 3d, 12h, 30m")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    args = parser.parse_args()

    if args.version:
        print(__version__)
        raise SystemExit(0)

    history_root = Path(args.history)
    workspace_dir = Path(args.workspace)

    if not history_root.exists():
        raise SystemExit(f"History directory not found: {history_root}")
    if not workspace_dir.exists():
        raise SystemExit(f"Workspace directory not found: {workspace_dir}")

    include_globs = [g.strip() for g in args.include.split(",") if g.strip()]
    exclude_globs = [g.strip() for g in args.exclude.split(",") if g.strip()]
    since_dt = parse_since(args.since) if args.since else None

    entries_paths = list(iter_entries_json_paths(history_root))

    statuses: List[FileStatus] = []

    def _submit(paths: List[Path]):
        if RICH:
            progress = Progress(SpinnerColumn(), "[progress.description]{task.description}", BarColumn(), TimeElapsedColumn())
            with progress:
                task = progress.add_task("Scanning history", total=len(paths))
                with cf.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                    futures = [pool.submit(process_history_dir, p, workspace_dir) for p in paths]
                    for fut in cf.as_completed(futures):
                        st = fut.result()
                        if st is not None:
                            statuses.append(st)
                        progress.advance(task)
        elif TQDM:
            with cf.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = [pool.submit(process_history_dir, p, workspace_dir) for p in paths]
                for fut in tqdm(cf.as_completed(futures), total=len(paths), desc="Scanning"):
                    st = fut.result()
                    if st is not None:
                        statuses.append(st)
        else:
            with cf.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = [pool.submit(process_history_dir, p, workspace_dir) for p in paths]
                for i, fut in enumerate(cf.as_completed(futures), 1):
                    st = fut.result()
                    if st is not None:
                        statuses.append(st)
                    if i % 200 == 0:
                        print(f"Scanned {i}/{len(paths)}…", file=sys.stderr)

    _submit(entries_paths)

    # Keep only items under include/exclude and since window, then keep DELETED/MODIFIED for the main report
    filtered: List[FileStatus] = []
    for st in statuses:
        if not matches_globs(st.resource_path, include_globs):
            continue
        if exclude_globs and any(st.resource_path.match(p) for p in exclude_globs):
            continue
        if not within_since(since_dt, st.last_edit_at):
            continue
        if st.status in {"DELETED", "MODIFIED"}:
            filtered.append(st)

    if args.restore_deleted:
        restore_results = restore_deleted_files(
            filtered,
            dry_run=args.restore_dry_run,
            overwrite_if_exists=args.restore_overwrite,
        )
        # Always emit machine-readable output alongside restore to keep logs
        if args.output == "json":
            print_json(filtered, include_diffs=args.diff)
        elif args.output == "csv":
            print_csv(filtered)
        elif args.output == "ndjson":
            print_ndjson(filtered)
        else:
            print_human(filtered, include_diffs=args.diff)
        print(json.dumps([dc.asdict(r) for r in restore_results], indent=2))
    else:
        if not filtered:
            msg = "No deleted or modified files detected under the specified workspace."
            if RICH:
                console.print(f":white_check_mark: [green]{msg}[/green]")
            else:
                print(msg)
            return
        if args.output == "json":
            print_json(filtered, include_diffs=args.diff)
        elif args.output == "csv":
            print_csv(filtered)
        elif args.output == "ndjson":
            print_ndjson(filtered)
        else:
            print_human(filtered, include_diffs=args.diff)

if __name__ == "__main__":
    main()
