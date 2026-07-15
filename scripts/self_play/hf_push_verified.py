#!/usr/bin/env python3
"""Upload a folder to a HuggingFace repo AND verify it actually landed.

The whole v6 loss happened because pushes silently failed and a cosmetic
"URLs:" echo gave false confidence. This never trusts the upload call — it
re-lists the repo afterwards and confirms the expected files are present,
exiting non-zero if not.

Usage:
    python3 scripts/self_play/hf_push_verified.py \
        --local <dir> --repo Abdine/<name> --repo-type model \
        --require config.json --require model.safetensors
    # or for LoRA adapters:  --require adapter_config.json
    # or for datasets:       --repo-type dataset --require <known file>

Exit codes: 0 = uploaded AND verified; 1 = failed / not verified.
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True, help="local folder to upload")
    ap.add_argument("--repo", required=True, help="Abdine/<name>")
    ap.add_argument("--repo-type", default="model",
                    choices=["model", "dataset"])
    ap.add_argument("--path-in-repo", default=".",
                    help="subpath inside the repo (default root)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--require", action="append", default=[],
                    help="filename(s) that MUST appear after upload; repeatable")
    ap.add_argument("--allow-patterns", default=None,
                    help="comma-sep globs to upload (default: everything)")
    ap.add_argument("--commit-message", default="upload")
    args = ap.parse_args()

    local = Path(args.local)
    if not local.is_dir():
        print(f"FAIL: local dir not found: {local}")
        return 1

    from huggingface_hub import HfApi
    api = HfApi()

    allow = None
    if args.allow_patterns:
        allow = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]

    try:
        api.create_repo(args.repo, repo_type=args.repo_type,
                        private=args.private, exist_ok=True)
        print(f"  repo ready: {args.repo} ({args.repo_type})")
        api.upload_folder(
            repo_id=args.repo, repo_type=args.repo_type,
            folder_path=str(local), path_in_repo=args.path_in_repo,
            allow_patterns=allow, commit_message=args.commit_message,
        )
        print(f"  upload call returned for {local} -> {args.repo}")
    except Exception as e:
        print(f"FAIL: upload error: {e!r}")
        return 1

    # ── VERIFY: re-list the repo and confirm required files are present ──────
    try:
        files = set(api.list_repo_files(args.repo, repo_type=args.repo_type))
    except Exception as e:
        print(f"FAIL: could not list repo to verify: {e!r}")
        return 1

    prefix = "" if args.path_in_repo in (".", "") else args.path_in_repo.rstrip("/") + "/"
    missing = []
    for req in args.require:
        # match either exact or under path-in-repo prefix, anywhere in tree
        hit = any(f == req or f == prefix + req or f.endswith("/" + req)
                  for f in files)
        if not hit:
            missing.append(req)

    if missing:
        print(f"FAIL: VERIFY — these required files are NOT on HF: {missing}")
        print(f"      (repo has {len(files)} files; push did NOT land properly)")
        return 1

    print(f"  VERIFIED: {len(files)} files on HF, all required present: "
          f"{args.require or '(no --require given)'}")
    print(f"  https://huggingface.co/{'datasets/' if args.repo_type=='dataset' else ''}{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
