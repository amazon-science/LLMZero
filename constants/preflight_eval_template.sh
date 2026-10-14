#!/bin/bash
set -x

# Preflight eval template — placeholders filled by the orchestrator
# {{DATASET_DIR}}       — path to dataset or run folder containing data + reward
# {{MODEL_PATH}}        — absolute path to the base model
# {{OUTPUT_DIR}}        — output directory for eval artifacts
# {{REWARD_FN_PATH}}    — path to compute_score.py
# {{REWARD_FN_NAME}}    — reward function name (usually compute_score)
# {{TRAIN_FILES}}       — VERL-format list of train parquet paths
# {{VAL_FILES}}         — VERL-format list of val/sample parquet paths
# {{PROJECT_NAME}}      — WandB project name
# {{EXPERIMENT_NAME}}   — WandB experiment name
# {{NNODES}}            — number of nodes
# {{N_GPUS_PER_NODE}}   — GPUs per node
# {{MAX_RESPONSE_LENGTH}} — max response length for inference
# {{INSTALL_COMMANDS}}  — pip install lines from install_commands.txt (or empty)

{{INSTALL_COMMANDS}}

python3 -m verl.trainer.main_ppo \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.logger='["console"]' \
    trainer.project_name={{PROJECT_NAME}} \
    trainer.experiment_name={{EXPERIMENT_NAME}} \
    trainer.nnodes={{NNODES}} \
    trainer.n_gpus_per_node={{N_GPUS_PER_NODE}} \
    trainer.resume_mode=disable \
    trainer.default_local_dir={{OUTPUT_DIR}} \
    trainer.validation_data_dir={{OUTPUT_DIR}} \
    hydra.run.dir={{OUTPUT_DIR}} \
    algorithm.adv_estimator=grpo \
    data.train_files="{{TRAIN_FILES}}" \
    data.val_files="{{VAL_FILES}}" \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length={{MAX_RESPONSE_LENGTH}} \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.name={{REWARD_FN_NAME}} \
    reward.custom_reward_function.path={{REWARD_FN_PATH}} \
    actor_rollout_ref.model.path={{MODEL_PATH}} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    2>&1 | tee {{OUTPUT_DIR}}/std_logs.txt
