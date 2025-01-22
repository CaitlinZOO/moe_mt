#!/usr/bin/bash

#SBATCH --job-name=moe-res-droppad-nosys-all
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

##SBATCH --nodes=4
##SBATCH --gres=gpu:8
#SBATCH --quotatype=reserved
#SBATCH -w SH-IDCA1404-10-140-54-[4,12,20,89]

# export WANDB_PROJECT="v2-mixtral"

{
    task_name=$SLURM_JOB_NAME
    model_type="mixtral2group"

    dataset_dir_or_path="/home/zhanglinlin/zll/moe_mt/mustc_fr.jsonl"
    model_name_or_path="/home/zhanglinlin/outputs/moe_mt/converted_models/Llama3.2-1B-2group-4-4expert-MLP-MoE-Top1-Scale4.0-Insert4"

    comment="Llama3.2-1B to mixtral-no-megablocks, 2group 4experts, top1"
    base_dir="/home/zhanglinlin/outputs/moe_mt/mixtral_2group"
    output_dir="${base_dir}/sft_lm/$SLURM_JOB_ID"
    mkdir -p $output_dir
    scontrol write batch_script $SLURM_JOBID $output_dir/sbatch.sh
    git diff > $output_dir/diff.patch
    env > $output_dir/env
    echo -e "Job ID: ${SLURM_JOB_ID}\n\nLog: logs/${task_name}-$SLURM_JOB_ID.log\n\nGit commit: $(git log -1 --oneline)\n\nGit branch: $(git branch | grep "*")\n\nComment: ${comment}" > $output_dir/comment.txt
    ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $output_dir/log.log
    echo "$SLURM_JOB_ID" > $base_dir/latest.jobid
    ln -snf $output_dir $base_dir/latest.dir
    ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $base_dir/latest.log

    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    echo "Node: $head_node"

    srun torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29522 \
        -m smoe.entrypoint.sft.train_sft_llama3_2group_lm.py \
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
            --model_max_length 4096 \
            --gradient_checkpointing True \
            --save_only_model True \

}
