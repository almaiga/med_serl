#!/usr/bin/env python3
"""Patch installed VERL files with debug prints for interaction selection.

This is a surgical helper for training boxes where custom multi-turn interactions
appear to be ignored silently. It instruments the installed VERL runtime to
print:

1. Which interactions were registered from the interaction config
2. Which interaction_kwargs were extracted for each sample
3. Which interaction name/class was selected before start_interaction()

The patch is idempotent.

The logging is intentionally unconditional. The current failure mode is that
env-gated debug prints never appeared even though the patched files were the
ones being imported, so the next step is to prove whether these code paths are
reached at all.
"""

from __future__ import annotations

import glob
from pathlib import Path


def _first_existing(*paths: str) -> Path | None:
    for raw in paths:
        p = Path(raw)
        if p.exists():
            return p
    return None


def _replace_once(code: str, target: str, replacement: str, desc: str) -> tuple[str, bool]:
    if replacement in code:
        return code, False
    if target not in code:
        print(f"WARNING: could not find target for {desc}")
        return code, False
    return code.replace(target, replacement, 1), True


def _resolve_tool_agent_loop() -> Path | None:
    p = _first_existing(
        "/sgl-workspace/sglang/verl/verl/experimental/agent_loop/tool_agent_loop.py",
        "/workspace/verl/verl/experimental/agent_loop/tool_agent_loop.py",
    )
    if p is not None:
        return p
    matches = glob.glob("/usr/local/lib/python*/dist-packages/verl/experimental/agent_loop/tool_agent_loop.py")
    return Path(matches[0]) if matches else None


def _resolve_interaction_registry() -> Path | None:
    p = _first_existing(
        "/sgl-workspace/sglang/verl/verl/interactions/utils/interaction_registry.py",
        "/workspace/verl/verl/interactions/utils/interaction_registry.py",
    )
    if p is not None:
        return p
    matches = glob.glob("/usr/local/lib/python*/dist-packages/verl/interactions/utils/interaction_registry.py")
    return Path(matches[0]) if matches else None


def patch_tool_agent_loop(path: Path) -> None:
    code = path.read_text()
    changed = False

    # Upgrade older env-gated MedSeRL debug prints to unconditional prints.
    old = (
        "        interaction_map = initialize_interactions_from_config(interaction_config_file)\n"
        "        if __import__('os').environ.get('MEDSERL_DEBUG_LOGGING', '0') == '1':\n"
        "            print(\n"
        "                f\"[MedSeRL][tool_agent_loop] interaction_config_file={interaction_config_file!r} \"\n"
        "                f\"interaction_map_keys={list(interaction_map.keys())}\",\n"
        "                flush=True,\n"
        "            )\n"
        "        return interaction_map\n"
    )
    new = (
        "        interaction_map = initialize_interactions_from_config(interaction_config_file)\n"
        "        print(\n"
        "            f\"[MedSeRL][tool_agent_loop] interaction_config_file={interaction_config_file!r} \"\n"
        "            f\"interaction_map_keys={list(interaction_map.keys())}\",\n"
        "            flush=True,\n"
        "        )\n"
        "        return interaction_map\n"
    )
    if old in code:
        code = code.replace(old, new, 1)
        changed = True

    old = (
        '            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]\n'
        "            if __import__('os').environ.get('MEDSERL_DEBUG_LOGGING', '0') == '1':\n"
        "                print(\n"
        "                    f\"[MedSeRL][tool_agent_loop] request_id={request_id!r} \"\n"
        "                    f\"interaction_kwargs={interaction_kwargs!r}\",\n"
        "                    flush=True,\n"
        "                )\n"
    )
    new = (
        '            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]\n'
        "            print(\n"
        "                f\"[MedSeRL][tool_agent_loop] request_id={request_id!r} \"\n"
        "                f\"interaction_kwargs={interaction_kwargs!r}\",\n"
        "                flush=True,\n"
        "            )\n"
    )
    if old in code:
        code = code.replace(old, new, 1)
        changed = True

    old = (
        "            if __import__('os').environ.get('MEDSERL_DEBUG_LOGGING', '0') == '1':\n"
        "                print(\n"
        "                    f\"[MedSeRL][tool_agent_loop] request_id={request_id!r} \"\n"
        "                    f\"interaction_name={interaction_name!r} interaction_cls={type(interaction).__module__}.{type(interaction).__name__}\",\n"
        "                    flush=True,\n"
        "                )\n"
        "            await interaction.start_interaction(request_id, **interaction_kwargs)\n"
    )
    new = (
        "            print(\n"
        "                f\"[MedSeRL][tool_agent_loop] request_id={request_id!r} \"\n"
        "                f\"interaction_name={interaction_name!r} interaction_cls={type(interaction).__module__}.{type(interaction).__name__}\",\n"
        "                flush=True,\n"
        "            )\n"
        "            await interaction.start_interaction(request_id, **interaction_kwargs)\n"
    )
    if old in code:
        code = code.replace(old, new, 1)
        changed = True

    target = "        interaction_map = initialize_interactions_from_config(interaction_config_file)\n        return interaction_map\n"
    replacement = (
        "        interaction_map = initialize_interactions_from_config(interaction_config_file)\n"
        "        print(\n"
        "            f\"[MedSeRL][tool_agent_loop] interaction_config_file={interaction_config_file!r} \"\n"
        "            f\"interaction_map_keys={list(interaction_map.keys())}\",\n"
        "            flush=True,\n"
        "        )\n"
        "        return interaction_map\n"
    )
    code, did = _replace_once(code, target, replacement, "tool_agent_loop interaction_map print")
    changed |= did

    target = '            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]\n'
    replacement = (
        '            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]\n'
        "            print(\n"
        "                f\"[MedSeRL][tool_agent_loop] request_id={request_id!r} \"\n"
        "                f\"interaction_kwargs={interaction_kwargs!r}\",\n"
        "                flush=True,\n"
        "            )\n"
    )
    code, did = _replace_once(code, target, replacement, "tool_agent_loop interaction_kwargs print")
    changed |= did

    target = "            await interaction.start_interaction(request_id, **interaction_kwargs)\n"
    replacement = (
        "            print(\n"
        "                f\"[MedSeRL][tool_agent_loop] request_id={request_id!r} \"\n"
        "                f\"interaction_name={interaction_name!r} interaction_cls={type(interaction).__module__}.{type(interaction).__name__}\",\n"
        "                flush=True,\n"
        "            )\n"
        "            await interaction.start_interaction(request_id, **interaction_kwargs)\n"
    )
    code, did = _replace_once(code, target, replacement, "tool_agent_loop selected interaction print")
    changed |= did

    if changed:
        path.write_text(code)
        print(f"PATCHED: {path}")
    else:
        print(f"UNCHANGED: {path}")


def patch_interaction_registry(path: Path) -> None:
    code = path.read_text()
    changed = False

    # Upgrade older env-gated MedSeRL debug prints to unconditional prints.
    old = (
        "    interaction_map = {}\n"
        "    if __import__('os').environ.get('MEDSERL_DEBUG_LOGGING', '0') == '1':\n"
        "        print(\n"
        "            f\"[MedSeRL][interaction_registry] config_file={interaction_config_file!r}\",\n"
        "            flush=True,\n"
        "        )\n"
    )
    new = (
        "    interaction_map = {}\n"
        "    print(\n"
        "        f\"[MedSeRL][interaction_registry] config_file={interaction_config_file!r}\",\n"
        "        flush=True,\n"
        "    )\n"
    )
    if old in code:
        code = code.replace(old, new, 1)
        changed = True

    old = (
        "        interaction_map[name] = interaction\n"
        "        if __import__('os').environ.get('MEDSERL_DEBUG_LOGGING', '0') == '1':\n"
        "            print(\n"
        "                f\"[MedSeRL][interaction_registry] registered name={name!r} \"\n"
        "                f\"interaction_cls={type(interaction).__module__}.{type(interaction).__name__}\",\n"
        "                flush=True,\n"
        "            )\n"
    )
    new = (
        "        interaction_map[name] = interaction\n"
        "        print(\n"
        "            f\"[MedSeRL][interaction_registry] registered name={name!r} \"\n"
        "            f\"interaction_cls={type(interaction).__module__}.{type(interaction).__name__}\",\n"
        "            flush=True,\n"
        "        )\n"
    )
    if old in code:
        code = code.replace(old, new, 1)
        changed = True

    target = "    interaction_map = {}\n"
    replacement = (
        "    interaction_map = {}\n"
        "    print(\n"
        "        f\"[MedSeRL][interaction_registry] config_file={interaction_config_file!r}\",\n"
        "        flush=True,\n"
        "    )\n"
    )
    code, did = _replace_once(code, target, replacement, "interaction_registry config print")
    changed |= did

    target = "        interaction_map[name] = interaction\n"
    replacement = (
        "        interaction_map[name] = interaction\n"
        "        print(\n"
        "            f\"[MedSeRL][interaction_registry] registered name={name!r} \"\n"
        "            f\"interaction_cls={type(interaction).__module__}.{type(interaction).__name__}\",\n"
        "            flush=True,\n"
        "        )\n"
    )
    code, did = _replace_once(code, target, replacement, "interaction_registry registration print")
    changed |= did

    if changed:
        path.write_text(code)
        print(f"PATCHED: {path}")
    else:
        print(f"UNCHANGED: {path}")


def main() -> int:
    tool_agent_loop = _resolve_tool_agent_loop()
    interaction_registry = _resolve_interaction_registry()

    if tool_agent_loop is None:
        print("SKIP: tool_agent_loop.py not found")
    else:
        patch_tool_agent_loop(tool_agent_loop)

    if interaction_registry is None:
        print("SKIP: interaction_registry.py not found")
    else:
        patch_interaction_registry(interaction_registry)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
