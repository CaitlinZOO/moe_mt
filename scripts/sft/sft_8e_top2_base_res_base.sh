#!/bin/bash
#!/usr/bin/env bash

#set -x

#expt_dir="$(dirname "$(readlink -f "$0")")"


ROOT=/home/zhanglinlin
cd ${ROOT}/pro/MoE/moe_mt
echo "进入 ${ROOT}/pro/MoE/moe_mt"  ## /mnt/alitranx-nas/users/zll240651/pro/LLM/blsp2

# additional Python packages for S2T data processing/model training
#pip install pandas torchaudio sentencepiece
echo " >>>>>>>>>>>>>>>>>>>>>>>>>>           "
nvidia-smi
echo " >>>>>>>>>>>>>>>>>>>>>>>>>>           "


export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export WANDB_DISABLED=true
export PATH=/usr/local/cuda/bin:$PATH
# export PATH=/usr/local/cuda-12.4/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# {
    task_name="tt" $SLURM_JOB_NAME
    model_type="v2_mixtral"

    dataset_dir_or_path="/home/zhanglinlin/.cache/modelscope/hub/datasets/580682a40c789fb45be034c24a2135258fa6e192a7457815bae60f7277dde6ab.json"
    model_name_or_path="/home/zhanglinlin/outputs/moe_mt/converted_models/Llama-3.2-1B-8expert-MLP-MoE-Top2-Scale4.0-Dense0"

    comment="llama-3-8b to mixtral-no-megablocks, 8 experts, top2"
    base_dir="/home/zhanglinlin/outputs/moe_mt/v2_mixtral"
    # output_dir="${base_dir}/${task_name}/$SLURM_JOB_ID"
    output_dir="${base_dir}/sft_conv/Llama-1B_8experts_top2-tt"
    mkdir -p $output_dir
    echo $output_dir
    # scontrol write batch_script $SLURM_JOBID $output_dir/sbatch.sh
    # git diff > $output_dir/diff.patch
    # env > $output_dir/env
    # echo -e "Job ID: ${SLURM_JOB_ID}\n\nLog: logs/${task_name}-$SLURM_JOB_ID.log\n\nGit commit: $(git log -1 --oneline)\n\nGit branch: $(git branch | grep "*")\n\nComment: ${comment}" > $output_dir/comment.txt
    # ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $output_dir/log.log
    # echo "$SLURM_JOB_ID" > $base_dir/latest.jobid
    # ln -snf $output_dir $base_dir/latest.dir
    # ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $base_dir/latest.log

    # nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
    # nodes_array=($nodes)
    # head_node=${nodes_array[0]}
    # echo "Node: $head_node"

    # srun torchrun \
    # --nnodes 4 \
    # --nproc_per_node 8 \
    # --node_rank $SLURM_NODEID \
    # --rdzv_id $RANDOM \
    # --rdzv_backend c10d \
    # --rdzv_endpoint $head_node:29522 \
    #     -m smoe.entrypoint.sft.train_sft_llama3_nopad \
    CUDA_VISIBLE_DEVICES=5 python smoe/entrypoint/sft/train_sft_llama3_nopad.py \
            --do_train \
            --freeze_gate False \
            --evaluation_strategy no \
            --run_name $task_name \
            --model_type $model_type \
            --model_name_or_path $model_name_or_path \
            --dataset_dir_or_path $dataset_dir_or_path \
            --output_dir $output_dir \
            --deepspeed conf/deepspeed/bf16_zero1.json \
            --seed 1227 \
            --bf16 True \
            --tf32 True \
            --torch_dtype bfloat16 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 2 \
            --num_train_epochs 3 \
            --save_strategy steps \
            --save_steps 1000 \
            --save_total_limit 10 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --logging_steps 10 \
            --model_max_length 1024 \
            --gradient_checkpointing True \
            --save_only_model True \

# }
