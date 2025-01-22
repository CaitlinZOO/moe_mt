#!/usr/bin/bash

#SBATCH --job-name=convert
#SBATCH --output=logs_split/%x-%j.log
#SBATCH --error=logs_split/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --quotatype=auto

{
  model_path="/home/zhanglinlin/models/llama/Meta-Llama-3.2-1B-Instruct"

  num_experts_group0=4
  num_experts_group1=4
  top_k=1
  scale_factor=4.0   # we suggest this value to be 4.0 for 8 experts
  num_moe_insert_layers=4   ## insert MoE experts per 4 layers
  # moe_implementation_type="modulelist"

  folder_name="group0-${num_experts_group0}experts-group1-${num_experts_group0}experts"
  split_folder_name="2group-${num_experts_group0}-${num_experts_group0}expert-MLP-MoE"
  save_folder_name="Llama3.2-1B-${split_folder_name}-Top${top_k}-Scale${scale_factor}-Insert${num_moe_insert_layers}"

  save_path="/home/zhanglinlin/outputs/moe_mt/converted_models/${save_folder_name}"

#  srun
  python smoe/utils/expert_construction/convert_llama_to_mixtral_2group.py \
    --model_path ${model_path} \
    --save_path ${save_path} \
    --num_experts_group0 ${num_experts_group0} \
    --num_experts_group1 ${num_experts_group1} \
    --top_k ${top_k} \
    --scale_factor ${scale_factor} \
    --num_moe_insert_layers ${num_moe_insert_layers}
}