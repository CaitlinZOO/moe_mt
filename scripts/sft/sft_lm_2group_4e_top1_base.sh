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

# src_text
# tgt_text  
# src_lang
# tgt_lang

    task_name=$SLURM_JOB_NAME
    model_type="mixtral2group"

    dataset_dir_or_path="/home/zhanglinlin/zll/en-es/dev_jst.tsv|/home/zhanglinlin/zll/en-fr/dev_jst.tsv"
    model_name_or_path="/home/zhanglinlin/outputs/moe_mt/converted_models/Llama3.2-1B-2group-4-4expert-MLP-MoE-Top1-Scale4.0-Insert4_use-fft"

    comment="Llama3.2-1B to mixtral-no-megablocks, 2group 4experts, top1"
    base_dir="/home/zhanglinlin/outputs/moe_mt/mixtral_2group"
    output_dir="${base_dir}/sft_lm/Llama-1B_2group_4experts_top1_fft-tt"
    data_dir=${output_dir}/data
    mkdir -p $output_dir $data_dir ${output_dir}/code
    cp -r ${ROOT}/pro/MoE/moe_mt/smoe ${output_dir}/code
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

GPUS="0,1,2,3"
ARR_GPU=(${GPUS//,/ })
NUM_GPU=${#ARR_GPU[@]}

echo "Request ${NUM_GPU} GPUs(${GPUS}) ."

export CUDA_VISIBLE_DEVICES=${GPUS}
port=$(( 104 + 26100 ))
# torchrun --nproc_per_node ${NUM_GPU} --master_port ${port}  
#  CUDA_VISIBLE_DEVICES=0,1 python 
    # --deepspeed config/dp_config_zero1.json \ --nnode=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR
# /home/zhanglinlin/anaconda3/envs/smoe/bin/python

# python -m torch.distributed.run --nproc_per_node=4 --nnode=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=${port} \
    CUDA_VISIBLE_DEVICES=4 python smoe/entrypoint/sft/train_sft_llama3_2group_st.py \
            --do_train \
            --freeze_gate False \
            --evaluation_strategy no \
            --model_type $model_type \
            --model_name_or_path $model_name_or_path \
            \
            --dataset_save_dir ${data_dir} \
            --remove_unused_columns False \
             --manifest_files ${dataset_dir_or_path} \
             --instructions "|" \
             --input_fields "src_text|src_text" \
             --output_fields "src_text|src_text" \
             --sample_probs "1|1" \
            \
            --output_dir $output_dir \
            --deepspeed conf/deepspeed/bf16_zero1.json \
            --seed 1227 \
            --bf16 True \
            \
            --torch_dtype bfloat16 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 2 \
            \
            --num_train_epochs 3 \
            --save_strategy steps \
            --save_steps 1000 \
            --save_total_limit 10 \
            \
            --learning_rate 2e-5 \
            --weight_decay 0.0 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --logging_steps 10 \
            --model_max_length 1024 \
            --gradient_checkpointing True \
            --save_only_model True   | tee -a ${output_dir}/train-bsz8-t0.log
    # --disable_tqdm True \
  ##--tf32 True \
# }
