#!/usr/bin/env python3
"""Patch verl's main_ppo.py to use Docker-safe Ray init defaults.

Handles three cases:
  a) Fresh main_ppo.py (never patched) — apply full patch
  b) Already patched but missing object_store_memory — add it
  c) Fully patched — no-op
"""
import pathlib

fpath = pathlib.Path("/workspace/verl/verl/trainer/main_ppo.py")
if not fpath.exists():
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
    print(f"ray init kwargs (patched): {_ray_kw}")
    ray.init(**_ray_kw)"""

    if "_ray_kw" not in code and fresh_target in code:
        # Case (a): never patched
        code = code.replace(fresh_target, full_replacement)
        fpath.write_text(code)
        print("PATCHED: main_ppo.py — full Docker-safe Ray defaults")
    elif "_ray_kw" in code and "object_store_memory" not in code:
        # Case (b): patched before but missing object_store_memory
        insert_before = '    print(f"ray init kwargs (patched):'
        new_lines = (
            "    _ray_kw.setdefault('object_store_memory', 1_000_000_000)  # 1 GB — safe for small /dev/shm\n"
            "    _ray_kw.setdefault('_plasma_directory', '/workspace/ray_tmp')  # bypass /dev/shm entirely\n"
        )
        if insert_before in code:
            code = code.replace(insert_before, new_lines + insert_before)
            fpath.write_text(code)
            print("PATCHED: added object_store_memory + _plasma_directory to existing patch")
        else:
            print("WARNING: could not find insertion point — manual check needed")
    else:
        print("main_ppo.py already fully patched")
