#!/usr/bin/env python3
"""Validate the archived legacy self_play.yaml against veRL 0.8.0.dev0 dataclass fields."""
import yaml, sys

with open('scripts/self_play/archive/legacy_agentic_umls/configs/self_play.yaml') as f:
    cfg = yaml.safe_load(f)

# 0.8.0.dev0 field inventories (from diagnostic output on pod)
FIELDS = {
    'verl.workers.config.TraceConfig': {
        '_target_','backend','token2text','max_samples_per_step_per_worker',
    },
    'verl.utils.profiler.ProfilerConfig': {
        '_target_','tool','enable','all_ranks','ranks','save_path',
        'tool_config','global_tool_config',
    },
    'verl.workers.config.RolloutConfig': {
        '_target_','name','mode','temperature','top_k','top_p',
        'prompt_length','response_length','dtype','gpu_memory_utilization',
        'ignore_eos','enforce_eager','cudagraph_capture_sizes','free_cache_engine',
        'tensor_model_parallel_size','data_parallel_size','expert_parallel_size',
        'pipeline_model_parallel_size','max_num_batched_tokens','max_model_len',
        'max_num_seqs','enable_chunked_prefill','enable_prefix_caching',
        'logprobs_mode','scheduling_policy','load_format',
        'log_prob_micro_batch_size','log_prob_micro_batch_size_per_gpu',
        'log_prob_use_dynamic_bsz','log_prob_max_token_len_per_gpu',
        'disable_log_stats','do_sample','n','over_sample_rate',
        'multi_stage_wake_up','engine_kwargs','val_kwargs','multi_turn',
        'calculate_log_probs','agent','checkpoint_engine','trace',
        'skip_rollout','skip_dump_dir','skip_tokenizer_init',
        'enable_rollout_routing_replay','profiler','prometheus',
        'quantization','quantization_config_file','mtp','qat',
        'layered_summon','server','layer_name_map','limit_images',
        'repetition_penalty','custom','sglang_engine_mode','enable_sleep_mode',
    },
    'verl.workers.config.FSDPActorConfig': {
        '_target_','strategy','rollout_n','ppo_mini_batch_size',
        'ppo_micro_batch_size','ppo_micro_batch_size_per_gpu',
        'use_dynamic_bsz','ppo_max_token_len_per_gpu','grad_clip',
        'clip_ratio','clip_ratio_low','clip_ratio_high','clip_ratio_c',
        'tau_pos','tau_neg','freeze_vision_tower','policy_loss',
        'loss_agg_mode','loss_scale_factor','entropy_coeff',
        'calculate_entropy','use_kl_loss','use_prefix_grouper',
        'use_torch_compile','kl_loss_coef','kl_loss_type','ppo_epochs',
        'shuffle','data_loader_seed','checkpoint','use_fused_kernels',
        'profiler','router_replay','ulysses_sequence_parallel_size',
        'entropy_from_logits_with_chunking','entropy_checkpointing',
        'use_remove_padding','calculate_sum_pi_squared',
        'sum_pi_squared_checkpointing','qat','optim','fsdp_config',
        'engine','model_config','global_batch_info',
        'ppo_infer_micro_batch_size_per_gpu',
        'ppo_infer_max_token_len_per_gpu','use_rollout_log_probs',
        'log_prob_micro_batch_size','log_prob_micro_batch_size_per_gpu',
        'log_prob_use_dynamic_bsz','log_prob_max_token_len_per_gpu',
    },
    'verl.workers.config.FSDPCriticConfig': {
        '_target_','enable','strategy','rollout_n','ppo_mini_batch_size',
        'ppo_micro_batch_size','ppo_micro_batch_size_per_gpu',
        'use_dynamic_bsz','ppo_max_token_len_per_gpu',
        'forward_max_token_len_per_gpu','ppo_epochs','shuffle',
        'data_loader_seed','cliprange_value','loss_agg_mode',
        'forward_micro_batch_size','forward_micro_batch_size_per_gpu',
        'ulysses_sequence_parallel_size','grad_clip','model','model_config',
        'optim','checkpoint','profiler','engine','use_remove_padding','qat',
    },
    'verl.workers.config.FSDPEngineConfig': {
        '_target_','param_offload','optimizer_offload','grad_offload',
        'forward_only','strategy','dtype','use_dynamic_bsz',
        'max_token_len_per_gpu','micro_batch_size_per_gpu',
        'infer_max_token_len_per_gpu','infer_micro_batch_size_per_gpu',
        'use_fused_kernels','use_remove_padding','seed','full_determinism',
        'router_replay','wrap_policy','offload_policy','reshard_after_forward',
        'fsdp_size','forward_prefetch','model_dtype','use_orig_params',
        'mixed_precision','ulysses_sequence_parallel_size',
        'entropy_from_logits_with_chunking','use_torch_compile',
        'entropy_checkpointing',
    },
    'verl.workers.config.HFModelConfig': {
        '_target_','path','hf_config_path','tokenizer_path','use_shm',
        'trust_remote_code','custom_chat_template','external_lib',
        'override_config','enable_gradient_checkpointing',
        'enable_activation_offload','use_remove_padding','lora_rank',
        'lora_alpha','target_modules','exclude_modules','lora_adapter_path',
        'use_liger','use_fused_kernels','fused_kernel_options','tiled_mlp','mtp',
        'local_path','local_hf_config_path','local_tokenizer_path',
        'load_tokenizer','hf_config','generation_config','tokenizer',
        'processor','target_parameters','lora','architectures','rope_scaling',
    },
    'verl.trainer.config.AlgoConfig': {
        '_target_','gamma','lam','adv_estimator','norm_adv_by_std_in_grpo',
        'use_kl_in_reward','kl_penalty','kl_ctrl','use_pf_ppo','pf_ppo',
        'rollout_correction','filter_groups',
    },
    'verl.trainer.config.CheckpointConfig': {
        '_target_','save_contents','load_contents','async_save','mbridge_config',
    },
    'verl.workers.config.FSDPOptimizerConfig': {
        '_target_','optimizer','optimizer_impl','lr','lr_warmup_steps_ratio',
        'total_training_steps','weight_decay','lr_warmup_steps','betas',
        'clip_grad','grad_clip','min_lr_ratio','num_cycles',
        'lr_scheduler_type','zero_indexed_step','warmup_style',
        'override_optimizer_config',
    },
    'verl.workers.config.PolicyLossConfig': {
        '_target_','loss_mode','clip_cov_ratio','clip_cov_lb','clip_cov_ub',
        'kl_cov_ratio','ppo_kl_coef',
    },
    'verl.workers.config.SamplingConfig': {
        '_target_','temperature','top_k','top_p','do_sample','n',
    },
    'verl.workers.config.MultiTurnConfig': {
        '_target_','enable','max_assistant_turns','tool_config_path',
        'max_user_turns','max_parallel_calls','max_tool_response_length',
        'tool_response_truncate_side','interaction_config_path',
        'use_inference_chat_template','tokenization_sanity_check_mode',
        'format','num_repeat_rollouts',
    },
    'verl.workers.config.AgentLoopConfig': {
        '_target_','num_workers','default_agent_loop',
        'agent_loop_config_path','custom_async_server',
        'agent_loop_manager_class',
    },
    'verl.workers.config.CustomAsyncServerConfig': {
        '_target_','path','name',
    },
    'verl.workers.config.CheckpointEngineConfig': {
        '_target_','backend','update_weights_bucket_megabytes','engine_kwargs',
    },
    'verl.workers.config.RouterReplayConfig': {
        '_target_','mode','record_file','replay_file',
    },
    'verl.trainer.config.KLControlConfig': {
        '_target_','type','kl_coef','horizon','target_kl',
    },
    # Profiler tool configs (sub-configs)
    'verl.utils.profiler.config.NsightToolConfig': {
        '_target_','discrete','controller_nsight_options','worker_nsight_options',
    },
    'verl.utils.profiler.config.NPUToolConfig': {
        '_target_','contents','level','analysis','discrete',
    },
    'verl.utils.profiler.config.TorchProfilerToolConfig': {
        '_target_','contents','discrete',
    },
    'verl.utils.profiler.config.TorchMemoryToolConfig': {
        '_target_','trace_alloc_max_entries','stack_depth',
    },
    # FSDPCriticModelCfg
    'verl.workers.config.FSDPCriticModelCfg': {
        '_target_','path','tokenizer_path','external_lib','trust_remote_code',
        'override_config','use_shm','enable_gradient_checkpointing',
        'enable_activation_offload','use_remove_padding','lora_rank',
        'lora_alpha','target_modules','fsdp_config','tiled_mlp',
        'lora_adapter_path','exclude_modules',
    },
}

errors = []

def check_node(node, path=''):
    if not isinstance(node, dict):
        return
    for k, v in node.items():
        child_path = f'{path}.{k}' if path else k
        if isinstance(v, dict):
            check_node(v, child_path)

    target = node.get('_target_')
    if not target or target not in FIELDS:
        return

    valid = FIELDS[target]
    for k in node:
        if k not in valid:
            errors.append(f'INVALID: {path}.{k} -- not in {target}')

check_node(cfg)

if errors:
    print(f'Found {len(errors)} invalid field(s):')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    print('All fields valid for veRL 0.8.0.dev0!')
