#!/bin/bash
#!/usr/bin/env bash

#set -x

#expt_dir="$(dirname "$(readlink -f "$0")")"


ROOT=/home/ubuntu   ## /home/zhanglinlin /home/ubuntu
cd ${ROOT}/pro/MoE/moe_mt
echo "进入 ${ROOT}/pro/MoE/moe_mt"  ## /data/zhanglinlin/pro/MoE/moe_mt

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




  model_path="${ROOT}/models/llama/Meta-Llama-3.2-1B-Instruct"

  num_experts_group0=2
  num_experts_group1=2
  top_k=1
  scale_factor=4.0   # we suggest this value to be 4.0 for 8 experts
  num_moe_insert_layers=4   ## insert MoE experts per 4 layers
  # moe_implementation_type="modulelist"

  folder_name="group0-${num_experts_group0}experts-group1-${num_experts_group0}experts"
  split_folder_name="2group-${num_experts_group0}-${num_experts_group0}expert-MLP-MoE"
  save_folder_name="Llama3.2-1B-${split_folder_name}-Top${top_k}-Scale${scale_factor}-Insert${num_moe_insert_layers}_use-fft" #_use-fft

  save_path="${ROOT}/outputs/moe_mt/converted_models/${save_folder_name}"



GPUS="0,1,2,3"
ARR_GPU=(${GPUS//,/ })
NUM_GPU=${#ARR_GPU[@]}

echo "Request ${NUM_GPU} GPUs(${GPUS}) ."

export CUDA_VISIBLE_DEVICES=${GPUS}
port=$(( 104 + 26300 ))
# torchrun --nproc_per_node ${NUM_GPU} --master_port ${port}
#  CUDA_VISIBLE_DEVICES=0,1 python
    # --deepspeed config/dp_config_zero1.json \ --nnode=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR
# /home/zhanglinlin/anaconda3/envs/smoe/bin/python

# python -m torch.distributed.run --nproc_per_node=4 --nnode=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=${port} \
# /home/zhanglinlin/anaconda3/envs/smoe/bin/python  train.py \
#     --deepspeed config/dp_config_zero1.json \
 python smoe/entrypoint/expert_construction/convert/convert_mixtral_2group.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --num_experts_group0 ${num_experts_group0} \
    --num_experts_group1 ${num_experts_group1} \
    --top_k ${top_k} \
    --scale_factor ${scale_factor} \
    --num_moe_insert_layers ${num_moe_insert_layers} \
    --use_fft True
