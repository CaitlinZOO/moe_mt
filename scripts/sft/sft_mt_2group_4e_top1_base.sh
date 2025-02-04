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


   
    model_type="mixtral2group"

    train_dataset_dir_or_path="/home/zhanglinlin/zll/en-es/dev_jst.tsv|/home/zhanglinlin/zll/en-fr/dev_jst.tsv"
    train_dataset_dir_or_path="/home/zhanglinlin/zll/mt/wmt18_x_en-t0.json"
    ## 实际没用到eval_data  那几个参数去掉，不设置就可以
    eval_dataset_dir_or_path="/home/zhanglinlin/zll/en-es/tst-COMMON_jst-t.tsv|/home/zhanglinlin/zll/en-es/tst-COMMON_jst-t.tsv"
    ## 需要第一阶段的model,    还要把几个tokenizer_*.json等 从转换的模型那儿复制到路径下
    model_name_or_path="/home/zhanglinlin/outputs/moe_mt/converted_models/Llama3.2-1B-2group-4-4expert-MLP-MoE-Top1-Scale4.0-Insert4_use-fft"
    
    mt_instructions="Please translate the following English text into Spanish: | Please translate the following English text into French: "
    echo ${mt_instructions}

    comment="Llama3.2-1B to mixtral-no-megablocks, 2group 4experts, top1"
    base_dir="/home/zhanglinlin/outputs/moe_mt/mixtral_2group"
    output_dir="${base_dir}/sft_mt/Llama-1B_2group_4experts_top1_fft-wm18"
    data_dir=${output_dir}/data
    mkdir -p $output_dir $data_dir ${output_dir}/code
    cp -r ${ROOT}/pro/MoE/moe_mt/smoe ${output_dir}/code
    echo $output_dir


GPUS="0,1,2,3"
ARR_GPU=(${GPUS//,/ })
NUM_GPU=${#ARR_GPU[@]}

echo "Request ${NUM_GPU} GPUs(${GPUS}) ."

export CUDA_VISIBLE_DEVICES=${GPUS}
port=$(( 104 + 26200 ))

# /home/zhanglinlin/anaconda3/envs/smoe/bin/python     CUDA_VISIBLE_DEVICES=4 python 

## 单机多卡设置
# python -m torch.distributed.run --nproc_per_node=4 --nnode=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=${port} \
    CUDA_VISIBLE_DEVICES=5 python smoe/entrypoint/sft/train_sft_llama3_2group_mt.py \
            --do_train \
            --freeze_gate False \
            --evaluation_strategy no \
            --model_type $model_type \
            --model_name_or_path $model_name_or_path \
            \
            --dataset_save_dir ${data_dir} \
            --remove_unused_columns False \
            --manifest_files ${train_dataset_dir_or_path} \
            --instructions "" \
            --instruction_fields "instruction" \
            --input_fields "input" \
            --output_fields "output" \
            --sample_probs "1" \
            --padding_side "left" \
            \
             --output_router_logits True \
             --use_layer_wise_balance True \
            \
            --output_dir $output_dir \
            --deepspeed conf/deepspeed/bf16_zero1.json \
            --seed 1227 \
            --bf16 True \
            \
            --torch_dtype bfloat16 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
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
            --disable_tqdm True \
            --eval_strategy steps \
            --eval_steps 100 \
            --model_max_length 1024 \
            --gradient_checkpointing True \
            --save_only_model True   | tee -a ${output_dir}/train-bsz8-t0.log
    # --disable_tqdm True \
  ##--tf32 True \
            # --eval_manifest_files ${eval_dataset_dir_or_path} \
            # --eval_instructions "${mt_instructions}" \
            # --eval_instruction_fields "|" \
            # --eval_input_fields "src_text|src_text" \
            # --eval_output_fields "tgt_text|tgt_text" \
            # --eval_sample_probs "1|1" \
# }
