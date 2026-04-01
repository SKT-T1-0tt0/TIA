#!/usr/bin/env python3
"""
Create GitHub release v1.0 and upload releases/TIA-v1.0.tar.gz.

Requires GITHUB_TOKEN (classic PAT with repo scope, or fine-grained with
Contents: Read and write on SKT-T1-0tt0/TIA).

Usage:
  export GITHUB_TOKEN=ghp_...
  python3 scripts/publish_release_v1.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OWNER = "SKT-T1-0tt0"
REPO = "TIA"
TAG = "v1.0"
ASSET_NAME = "TIA-v1.0.tar.gz"
API = "https://api.github.com"
API_VER = "2022-11-28"


def _headers(extra: dict | None = None) -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("错误: 请设置环境变量 GITHUB_TOKEN（GitHub PAT，需 repo / Contents 写权限）", file=sys.stderr)
        sys.exit(1)
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VER,
        "Authorization": f"Bearer {token}",
        "User-Agent": "TIA-publish-release-script",
    }
    if extra:
        h.update(extra)
    return h


def _request(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    json_body: dict | None = None,
    headers: dict | None = None,
) -> tuple[int, dict | list | str | None]:
    if json_body is not None:
        data = json.dumps(json_body).encode()
        hdr = _headers({"Content-Type": "application/json"})
    else:
        hdr = _headers()
    if headers:
        hdr.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdr, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
            code = resp.getcode()
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return e.code, json.loads(body) if body.strip().startswith("{") else body
    if not body:
        return code, None
    try:
        return code, json.loads(body)
    except json.JSONDecodeError:
        return code, body


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    asset_path = root / "releases" / ASSET_NAME
    if not asset_path.is_file():
        print(f"错误: 找不到 {asset_path}", file=sys.stderr)
        sys.exit(1)

    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "(unknown)"

    notes = (
        f"源码快照（`git archive`，前缀 `TIA-1.0/`），对应提交 `{sha}`。\n\n"
        "不含 `.git` 与大文件忽略项；与仓库 `main` 上该提交树一致。"
    )

    rel_url = f"{API}/repos/{OWNER}/{REPO}/releases/tags/{TAG}"
    code, rel = _request("GET", rel_url)

    if code == 404:
        create_url = f"{API}/repos/{OWNER}/{REPO}/releases"
        code, rel = _request(
            "POST",
            create_url,
            json_body={
                "tag_name": TAG,
                "name": TAG,
                "body": notes,
                "draft": False,
                "prerelease": False,
                "target_commitish": sha,
            },
        )
        if code not in (200, 201):
            print(f"创建 Release 失败 HTTP {code}: {rel}", file=sys.stderr)
            sys.exit(1)
    elif code != 200:
        print(f"查询 Release 失败 HTTP {code}: {rel}", file=sys.stderr)
        sys.exit(1)

    assert isinstance(rel, dict)
    upload_tmpl = rel.get("upload_url", "")
    if "{?name,label}" not in upload_tmpl:
        print(f"异常 upload_url: {upload_tmpl}", file=sys.stderr)
        sys.exit(1)
    upload_url = upload_tmpl.replace("{?name,label}", f"?name={ASSET_NAME}")

    for a in rel.get("assets") or []:
        if a.get("name") == ASSET_NAME:
            aid = a.get("id")
            if aid is not None:
                del_url = f"{API}/repos/{OWNER}/{REPO}/releases/assets/{aid}"
                dc, _ = _request("DELETE", del_url)
                if dc not in (200, 204):
                    print(f"删除旧附件失败 HTTP {dc}", file=sys.stderr)
                    sys.exit(1)
            break

    raw = asset_path.read_bytes()
    code, _ = _request(
        "POST",
        upload_url,
        data=raw,
        headers={"Content-Type": "application/gzip"},
    )
    if code not in (200, 201):
        print(f"上传附件失败 HTTP {code}", file=sys.stderr)
        sys.exit(1)

    html = rel.get("html_url") or f"https://github.com/{OWNER}/{REPO}/releases/tag/{TAG}"
    print(f"完成: {html}")
    print(f"附件: {ASSET_NAME} ({len(raw) // 1024 // 1024} MiB)")


if __name__ == "__main__":
    main()
