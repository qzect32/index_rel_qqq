from __future__ import annotations

import json
import subprocess
from pathlib import Path
import datetime as dt

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def sh(*args: str) -> str:
    r = subprocess.run(list(args), cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "").strip())
    return (r.stdout or "").strip()


def git_commits_24h() -> int:
    out = sh("git", "log", "--since=24 hours ago", "--oneline")
    if not out:
        return 0
    return len([ln for ln in out.splitlines() if ln.strip()])


def git_numstat_24h() -> tuple[int, int, int]:
    out = sh("git", "log", "--since=24 hours ago", "--pretty=tformat:", "--numstat")
    ins = dels = 0
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        a, b = parts[0], parts[1]
        if a.isdigit():
            ins += int(a)
        if b.isdigit():
            dels += int(b)
    return ins, dels, ins - dels


def git_files_changed_24h() -> int:
    out = sh("git", "log", "--since=24 hours ago", "--name-only", "--pretty=format:")
    files = {ln.strip() for ln in out.splitlines() if ln.strip()}
    return len(files)


def parse_ts(obj: dict) -> dt.datetime | None:
    ts = obj.get("_updated_at") or obj.get("_received_at")
    if not ts:
        return None
    if isinstance(ts, str):
        s = ts
    else:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return dt.datetime.fromisoformat(s)
    except Exception:
        try:
            return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
        except Exception:
            return None


def decisions_24h() -> tuple[int, int]:
    p = DATA / "decisions_log.jsonl"
    if not p.exists():
        return 0, 0
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)

    unique: set[str] = set()
    total = 0

    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        ts = parse_ts(obj)
        if not ts or ts < since:
            continue
        cats = obj.get("categories")
        if not isinstance(cats, dict):
            continue
        for cat, payload in cats.items():
            if not isinstance(payload, dict):
                continue
            for k in payload.keys():
                if k.endswith("_notes") or k == "notes":
                    continue
                unique.add(f"{cat}.{k}")
                total += 1

    return len(unique), total


def todo_progress() -> tuple[int, int, int, int, float]:
    p = REPO / "TODO_STATUS.md"
    if not p.exists():
        return 0, 0, 0, 0, 0.0
    total = done = inprog = blocked = 0
    cur = None
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ln.startswith("## "):
            if cur is not None:
                s = cur
                if s == "DONE":
                    done += 1
                elif s == "IN-PROGRESS":
                    inprog += 1
                elif s == "BLOCKED":
                    blocked += 1
            total += 1
            cur = "IN-PROGRESS"
            continue
        if ln.strip().startswith("- STATUS:") and cur is not None:
            cur = ln.strip().split(":", 1)[1].strip().upper()
    if cur is not None:
        s = cur
        if s == "DONE":
            done += 1
        elif s == "IN-PROGRESS":
            inprog += 1
        elif s == "BLOCKED":
            blocked += 1

    credit = float(done) + 0.35 * float(inprog)
    pct = (credit / float(total) * 100.0) if total else 0.0
    return total, done, inprog, blocked, round(pct, 1)


def main() -> int:
    commits = git_commits_24h()
    ins, dels, net = git_numstat_24h()
    files = git_files_changed_24h()
    d_unique, d_total = decisions_24h()
    todo_total, todo_done, todo_inprog, todo_blocked, todo_pct = todo_progress()

    print("commits_24h", commits)
    print("loc_insertions_24h", ins)
    print("loc_deletions_24h", dels)
    print("loc_net_24h", net)
    print("files_changed_24h", files)
    print("decisions_unique_keys_24h", d_unique)
    print("decisions_total_answers_24h", d_total)
    print("todo_total", todo_total)
    print("todo_done", todo_done)
    print("todo_in_progress", todo_inprog)
    print("todo_blocked", todo_blocked)
    print("todo_pct_norm", todo_pct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
