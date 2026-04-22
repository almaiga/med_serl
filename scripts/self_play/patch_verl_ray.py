#!/usr/bin/env python3
"""Patch verl runtime files for MedSeRL.

Patches applied:
  1. main_ppo.py — Docker-safe Ray init defaults + MEDSERL_GAME_LOG in runtime_env
  2. tool_agent_loop.py — store interaction info dict in agent_data.extra_fields
     so interaction_reward_passthrough can read assessor_label/phase via tool_extra_fields
  3. agent_loop.py — allow custom agent loops to emit per-token rm_scores via
     AgentLoopOutput.extra_fields["token_level_scores"]
"""
import os
import pathlib
import glob


def _first_existing(*paths: str) -> pathlib.Path | None:
    for raw in paths:
        p = pathlib.Path(raw)
        if p.exists():
            return p
    return None

# ── Patch 1: main_ppo.py ─────────────────────────────────────────────────────
fpath = _first_existing(
    "/sgl-workspace/sglang/verl/verl/trainer/main_ppo.py",
    "/workspace/verl/verl/trainer/main_ppo.py",
)
if fpath is None or not fpath.exists():
    print("SKIP: main_ppo.py not found")
else:
    code = fpath.read_text()
    fresh_target = "ray.init(**OmegaConf.to_container(ray_init_kwargs))"
    full_replacement = """_ray_kw = OmegaConf.to_container(ray_init_kwargs)
    # ── MedSeRL Docker fix ──
    import os as _os
    if _ray_kw.get('num_cpus') is None:
        _ray_kw['num_cpus'] = min(_os.cpu_count() or 4, 8)
    _ray_kw.setdefault('include_dashboard', False)
    _ray_kw.setdefault('_temp_dir', '/workspace/ray_tmp')
    _ray_kw.setdefault('_node_ip_address', '127.0.0.1')
    _ray_kw.setdefault('object_store_memory', 1_000_000_000)  # 1 GB — safe for small /dev/shm
    _ray_kw.setdefault('_plasma_directory', '/workspace/ray_tmp')  # bypass /dev/shm entirely
    # Pass MEDSERL_GAME_LOG to Ray workers via runtime_env
    _game_log = _os.environ.get('MEDSERL_GAME_LOG', '')
    if _game_log:
        _re = _ray_kw.setdefault('runtime_env', {})
        _re.setdefault('env_vars', {})['MEDSERL_GAME_LOG'] = _game_log
    print(f"ray init kwargs (patched): {_ray_kw}")
    ray.init(**_ray_kw)"""

    if "_ray_kw" not in code and fresh_target in code:
        # Case (a): never patched
        code = code.replace(fresh_target, full_replacement)
        fpath.write_text(code)
        print("PATCHED: main_ppo.py — full Docker-safe Ray defaults + MEDSERL_GAME_LOG")
    elif "_ray_kw" in code and "MEDSERL_GAME_LOG" not in code:
        # Case (b): patched before but missing MEDSERL_GAME_LOG runtime_env
        insert_before = '    print(f"ray init kwargs (patched):'
        new_lines = (
            "    # Pass MEDSERL_GAME_LOG to Ray workers via runtime_env\n"
            "    _game_log = _os.environ.get('MEDSERL_GAME_LOG', '')\n"
            "    if _game_log:\n"
            "        _re = _ray_kw.setdefault('runtime_env', {})\n"
            "        _re.setdefault('env_vars', {})['MEDSERL_GAME_LOG'] = _game_log\n"
        )
        if insert_before in code:
            code = code.replace(insert_before, new_lines + insert_before)
            fpath.write_text(code)
            print("PATCHED: main_ppo.py — added MEDSERL_GAME_LOG runtime_env")
        else:
            print("WARNING: main_ppo.py — could not find insertion point for MEDSERL_GAME_LOG")
    else:
        print("main_ppo.py already fully patched")

# ── Patch 2: tool_agent_loop.py ──────────────────────────────────────────────
# veRL's ToolAgentLoop._handle_interacting_state receives the info dict from
# interaction.generate_response() as `metrics` but never stores it.
# This means interaction_reward_passthrough never sees phase/assessor_label.
# Fix: update agent_data.extra_fields with the info dict after each interaction turn.
tpath = _first_existing(
    "/sgl-workspace/sglang/verl/verl/experimental/agent_loop/tool_agent_loop.py",
    "/workspace/verl/verl/experimental/agent_loop/tool_agent_loop.py",
)
if tpath is None or not tpath.exists():
    # Try site-packages location
    matches = glob.glob("/usr/local/lib/python*/dist-packages/verl/experimental/agent_loop/tool_agent_loop.py")
    if matches:
        tpath = pathlib.Path(matches[0])
    else:
        print("SKIP: tool_agent_loop.py not found")
        tpath = None

if tpath and tpath.exists():
    tcode = tpath.read_text()
    target = "        if reward is not None:\n            agent_data.turn_scores.append(reward)\n"
    replacement = (
        "        if reward is not None:\n"
        "            agent_data.turn_scores.append(reward)\n"
        "        # MedSeRL: store interaction info dict so reward function can read it\n"
        "        if isinstance(metrics, dict) and metrics:\n"
        "            agent_data.extra_fields.update(metrics)\n"
    )
    if "MedSeRL: store interaction info dict" in tcode:
        print("tool_agent_loop.py already patched")
    elif target in tcode:
        tcode = tcode.replace(target, replacement)
        tpath.write_text(tcode)
        print(f"PATCHED: {tpath} — store interaction info dict in extra_fields")
    else:
        print(f"WARNING: tool_agent_loop.py — could not find patch target (manual check needed)")

# ── Patch 3: agent_loop.py ───────────────────────────────────────────────────
apath = _first_existing(
    "/sgl-workspace/sglang/verl/verl/experimental/agent_loop/agent_loop.py",
    "/workspace/verl/verl/experimental/agent_loop/agent_loop.py",
)
if apath is None or not apath.exists():
    matches = glob.glob("/usr/local/lib/python*/dist-packages/verl/experimental/agent_loop/agent_loop.py")
    if matches:
        apath = pathlib.Path(matches[0])
    else:
        print("SKIP: agent_loop.py not found")
        apath = None

if apath and apath.exists():
    acode = apath.read_text()
    old_block = """        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores
"""
    new_block = """        token_level_scores = [input.extra_fields.get("token_level_scores") for input in inputs]
        if any(score is not None for score in token_level_scores):
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            for idx, score_list in enumerate(token_level_scores):
                if score_list is None:
                    continue
                if hasattr(score_list, "tolist"):
                    score_list = score_list.tolist()
                if not isinstance(score_list, (list, tuple)):
                    continue
                width = min(len(score_list), rm_scores.size(1))
                if width > 0:
                    rm_scores[idx, :width] = torch.tensor(score_list[:width], dtype=torch.float32)
            batch["rm_scores"] = rm_scores
        else:
            scores = [input.reward_score for input in inputs]
            if all(score is not None for score in scores):
                prompt_length = prompt_ids.size(1)
                response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
                rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
                rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
                batch["rm_scores"] = rm_scores
"""
    if 'token_level_scores = [input.extra_fields.get("token_level_scores") for input in inputs]' in acode:
        print("agent_loop.py already patched")
    elif old_block in acode:
        acode = acode.replace(old_block, new_block)
        apath.write_text(acode)
        print(f"PATCHED: {apath} — enable token-level rm_scores from agent loop extra_fields")
    else:
        print(f"WARNING: agent_loop.py — could not find rm_scores block (manual check needed)")
