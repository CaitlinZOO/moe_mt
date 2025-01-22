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




  model_path="/home/zhanglinlin/models/llama/Meta-Llama-3.2-1B-Instruct"
  # model_path="/mnt/petrelfs/share_data/quxiaoye/models/Meta-Llama-3-8B-Instruct"

  num_experts=8
  top_k=2
  scale_factor=4.0   # we suggest this value to be 4.0 for 8 experts
  num_moe_contract_layers=0
  moe_implementation_type="modulelist"

  folder_name="${num_experts}experts"
  split_folder_name="${num_experts}expert-MLP-MoE"
  save_folder_name="Llama-3.2-1B-${split_folder_name}-Top${top_k}-Scale${scale_factor}-Dense${num_moe_contract_layers}"

  save_path="/home/zhanglinlin/outputs/moe_mt/converted_models/${save_folder_name}"

 python smoe/entrypoint/expert_construction/convert/convert_mixtral_v2.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --num_experts ${num_experts} \
    --top_k ${top_k} \
    --scale_factor ${scale_factor} \
    --num_moe_contract_layers ${num_moe_contract_layers} \
    --moe_implementation_type ${moe_implementation_type}
# }