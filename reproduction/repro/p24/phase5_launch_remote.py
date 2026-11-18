"""Run a single registered command and retain its process completion receipt."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    started = datetime.now(timezone.utc).isoformat()
    with args.log.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        result = process.wait()
    with args.receipt.open("x") as stream:
        json.dump({"pid": process.pid, "wrapper_pid": os.getpid(), "command": command, "started_at": started,
                   "finished_at": datetime.now(timezone.utc).isoformat(), "returncode": result}, stream, indent=2)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
