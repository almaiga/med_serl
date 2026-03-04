#!/usr/bin/env python3
"""Dump all veRL dataclass fields for version compatibility check."""
import dataclasses

CLASSES = [
    ("HFModelConfig", "verl.workers.config", "HFModelConfig"),
    ("FSDPActorConfig", "verl.workers.config", "FSDPActorConfig"),
    ("FSDPEngineConfig", "verl.workers.config", "FSDPEngineConfig"),
    ("FSDPOptimizerConfig", "verl.workers.config", "FSDPOptimizerConfig"),
    ("RolloutConfig", "verl.workers.config", "RolloutConfig"),
    ("FSDPCriticConfig", "verl.workers.config", "FSDPCriticConfig"),
    ("FSDPCriticModelCfg", "verl.workers.config", "FSDPCriticModelCfg"),
    ("CheckpointConfig", "verl.trainer.config", "CheckpointConfig"),
    ("AlgoConfig", "verl.trainer.config", "AlgoConfig"),
    ("ProfilerConfig", "verl.utils.profiler", "ProfilerConfig"),
    ("TraceConfig", "verl.workers.config", "TraceConfig"),
    ("AgentLoopConfig", "verl.workers.config", "AgentLoopConfig"),
    ("MultiTurnConfig", "verl.workers.config", "MultiTurnConfig"),
    ("SamplingConfig", "verl.workers.config", "SamplingConfig"),
    ("PrometheusConfig", "verl.workers.config", "PrometheusConfig"),
    ("CheckpointEngineConfig", "verl.workers.config", "CheckpointEngineConfig"),
    ("CustomAsyncServerConfig", "verl.workers.config", "CustomAsyncServerConfig"),
    ("RouterReplayConfig", "verl.workers.config", "RouterReplayConfig"),
    ("PolicyLossConfig", "verl.workers.config", "PolicyLossConfig"),
    ("KLControlConfig", "verl.trainer.config", "KLControlConfig"),
    ("MtpConfig", "verl.workers.config", "MtpConfig"),
    ("RewardManagerConfig", "verl.workers.config.reward_model", "RewardManagerConfig"),
    ("ModuleConfig", "verl.trainer.config.config", "ModuleConfig"),
]

# Also check profiler tool configs
TOOL_CONFIGS = [
    ("NsightToolConfig", "verl.utils.profiler.config", "NsightToolConfig"),
    ("NPUToolConfig", "verl.utils.profiler.config", "NPUToolConfig"),
    ("TorchProfilerToolConfig", "verl.utils.profiler.config", "TorchProfilerToolConfig"),
    ("TorchMemoryToolConfig", "verl.utils.profiler.config", "TorchMemoryToolConfig"),
]

def dump_class(label, mod_name, cls_name):
    try:
        mod = __import__(mod_name, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        fields = dataclasses.fields(cls)
        print(f"=== {label} ({len(fields)} fields) ===")
        for f in fields:
            has_default = (f.default is not dataclasses.MISSING or
                          f.default_factory is not dataclasses.MISSING)
            tag = "OPT" if has_default else "REQ"
            print(f"  {f.name}: {tag}")
    except Exception as e:
        print(f"=== {label}: ERROR: {e} ===")
    print()

for label, mod_name, cls_name in CLASSES:
    dump_class(label, mod_name, cls_name)

print("--- PROFILER TOOL CONFIGS ---")
for label, mod_name, cls_name in TOOL_CONFIGS:
    dump_class(label, mod_name, cls_name)
