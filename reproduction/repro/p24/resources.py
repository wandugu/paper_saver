"""Download pinned P24 resources and record byte-level provenance."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse


MODEL_REPOSITORIES = {
    "text": ("answerdotai/ModernBERT-base", "ModernBERT-base"),
    "vision": ("google/siglip2-base-patch16-naflex", "siglip2-base-patch16-naflex"),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    entries = [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
         "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(root).parts
        and not path.name.endswith(".partial")
    ]
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['path']}\0{entry['sha256']}\n".encode())
    return {"root": str(root), "files": entries, "tree_sha256": digest.hexdigest()}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download_model(repo_id: str, destination: str | Path, *,
                   revision: str = "main", endpoint: str | None = None,
                   workers: int = 3, weights_sha256: str | None = None) -> dict[str, Any]:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    endpoint = (endpoint or os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")).rstrip("/")
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def session() -> requests.Session:
        client = requests.Session()
        client.trust_env = False
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            client.proxies.update({"https": proxy, "http": proxy})
        client.mount("https://", HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))))
        return client

    response = session().get(f"{endpoint}/api/models/{repo_id}/revision/{revision}",
                             params={"blobs": "true"}, timeout=(15, 90))
    response.raise_for_status()
    metadata = response.json()
    immutable_revision = metadata["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", immutable_revision):
        raise ValueError(f"Expected immutable git revision, got {immutable_revision!r}")
    if re.fullmatch(r"[0-9a-f]{40}", revision) and immutable_revision != revision:
        raise ValueError(f"Resolved revision differs from the requested commit for {repo_id}")
    siblings = [entry for entry in metadata["siblings"]
                if not entry["rfilename"].startswith(("onnx/", "openvino/"))
                and not entry["rfilename"].endswith((".bin", ".h5", ".msgpack", ".ot"))
                and entry["rfilename"] != ".gitattributes"]
    if weights_sha256 is not None:
        weights = [entry.get("lfs", {}).get("sha256") for entry in siblings
                   if entry["rfilename"] == "model.safetensors"]
        if weights != [weights_sha256]:
            raise ValueError(f"Published weights differ from the locked SHA-256 for {repo_id}")

    def fetch(entry: dict[str, Any]) -> dict[str, Any]:
        relative = Path(entry["rfilename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe repository filename: {relative}")
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        expected = entry.get("lfs", {}).get("sha256")
        expected_bytes = entry.get("size")
        if output.exists() and expected and sha256_file(output) == expected:
            return {"path": str(relative), "bytes": output.stat().st_size,
                    "sha256": expected, "verified_against_lfs": True}
        partial = output.with_name(output.name + ".partial")
        for attempt in range(4):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                url = f"{endpoint}/{repo_id}/resolve/{immutable_revision}/{relative.as_posix()}"
                download = session().get(url, headers=headers, stream=True, allow_redirects=False,
                                         params={"download": "true", "p24": time.time_ns()},
                                         timeout=(20, 120))
                download.raise_for_status()
                transport: dict[str, Any] = {"mode": "https"}
                if download.is_redirect:
                    redirected = download.headers["location"]
                    host = urlparse(redirected).hostname
                    command = ["curl", "--fail", "--silent", "--show-error",
                               "--retry", "2", "--max-time", "600", "--connect-timeout", "20"]
                    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
                    if proxy:
                        command.extend(["--proxy", proxy])
                    if host == "cas-bridge.xethub.hf.co":
                        resolver = "https://dns.alidns.com/resolve"
                        answer = session().get(resolver, params={"name": host, "type": "A"},
                                               timeout=(10, 30))
                        answer.raise_for_status()
                        addresses = [entry["data"] for entry in answer.json().get("Answer", [])
                                     if entry.get("type") == 1]
                        if not addresses:
                            raise ValueError(f"No public DNS address for {host}")
                        command.extend(["--connect-to", f"{host}:443:{addresses[0]}:443"])
                        transport = {"mode": "curl_connect_to", "dns_resolver": resolver,
                                     "host": host, "address": addresses[0],
                                     "tls_hostname_verification": True}
                    if offset:
                        command.extend(["--continue-at", "-"])
                    command.extend(["--output", str(partial), redirected])
                    completed = subprocess.run(command, capture_output=True, timeout=660)
                    if completed.returncode:
                        raise OSError(completed.stderr.decode(errors="replace")[:1000])
                else:
                    mode = "ab" if offset and download.status_code == 206 else "wb"
                    with partial.open(mode) as stream:
                        for chunk in download.iter_content(4 * 1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                actual_hash = sha256_file(partial)
                if expected and actual_hash != expected:
                    raise ValueError(f"LFS hash mismatch for {relative}")
                if expected_bytes is not None and partial.stat().st_size != expected_bytes:
                    raise ValueError(f"File size mismatch for {relative}")
                partial.replace(output)
                print(f"downloaded {repo_id} {relative} {output.stat().st_size}", flush=True)
                return {"path": str(relative), "bytes": output.stat().st_size,
                        "sha256": actual_hash, "verified_against_lfs": bool(expected),
                        "transport": transport}
            except (requests.RequestException, OSError):
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Download failed: {relative}")

    print(f"resolved {repo_id}@{immutable_revision}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(fetch, siblings))
    record = {
        "status": "downloaded", "repository": repo_id,
        "upstream": f"https://huggingface.co/{repo_id}", "download_endpoint": endpoint,
        "requested_revision": revision, "revision": immutable_revision,
        "root": str(destination), "replacement": None,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(files, key=lambda entry: entry["path"]),
        "omitted_formats": ["PyTorch pickle weights", "Flax", "TensorFlow", "ONNX", "OpenVINO"],
    }
    write_json(destination / "p24_resource_manifest.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", default=os.environ.get("MODEL_ROOT", "models"))
    parser.add_argument("--inventory", default="data/p24/resource_inventory.json")
    parser.add_argument("--model", choices=("all", "text", "vision"), default="all")
    args = parser.parse_args()
    selected = MODEL_REPOSITORIES if args.model == "all" else {
        args.model: MODEL_REPOSITORIES[args.model]}
    lock = json.loads(Path(__file__).with_name("resources.lock.json").read_text(encoding="utf-8"))["models"]
    for key, (repo, _) in selected.items():
        if lock[key]["repository"] != repo:
            raise ValueError(f"Model repository differs from the resource lock for {key}")
    records: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        pending = {
            pool.submit(download_model, repo, Path(args.models_root).expanduser() / name,
                        revision=lock[key]["revision"], weights_sha256=lock[key]["weights_sha256"]): key
            for key, (repo, name) in selected.items()
        }
        for future in concurrent.futures.as_completed(pending):
            key = pending[future]
            try:
                records[key] = future.result()
            except Exception as error:
                records[key] = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                                "repository": selected[key][0]}
    inventory_path = Path(args.inventory)
    inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else {}
    inventory.setdefault("models", {}).update(records)
    inventory["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(inventory_path, inventory)
    if any(record["status"] == "failed" for record in records.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
