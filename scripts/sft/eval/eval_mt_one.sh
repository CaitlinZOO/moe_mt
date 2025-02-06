
ROOT=/home/zhanglinlin
cd ${ROOT}/pro/MoE/moe_mt
echo "进入 ${ROOT}/pro/MoE/moe_mt" 

CUDA_VISIBLE_DEVICES=2 /home/zhanglinlin/anaconda3/envs/smoe_ori/bin/python smoe/entrypoint/sft/test_generate_one.py
