
ROOT=/home/zhanglinlin
cd ${ROOT}/pro/MoE/moe_mt
echo "进入 ${ROOT}/pro/MoE/moe_mt"

output_path=/home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0
mkdir -p $output_path
touch $output_path/result.txt

CUDA_VISIBLE_DEVICES=0 nohup /home/zhanglinlin/anaconda3/envs/smoe_ori/bin/python smoe/entrypoint/sft/test_generate_one.py > /home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0/cs.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup /home/zhanglinlin/anaconda3/envs/smoe_ori/bin/python smoe/entrypoint/sft/test_generate_one-1.py > /home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0/de.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup /home/zhanglinlin/anaconda3/envs/smoe_ori/bin/python smoe/entrypoint/sft/test_generate_one-2.py > /home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0/et.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup /home/zhanglinlin/anaconda3/envs/smoe_ori/bin/python smoe/entrypoint/sft/test_generate_one-3.py > /home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0/fi.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup /home/zhanglinlin/anaconda3/envs/smoe_ori/bin/python smoe/entrypoint/sft/test_generate_one-4.py > /home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0/ru.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup /home/zhanglinlin/anaconda3/envs/smoe_ori/bin/python smoe/entrypoint/sft/test_generate_one-5.py > /home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0/tr.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup /home/zhanglinlin/anaconda3/envs/smoe_ori/bin/python smoe/entrypoint/sft/test_generate_one-6.py > /home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0/zh.log 2>&1 &


/home/zhanglinlin/anaconda3/envs/smoe_ori/bin/python smoe/entrypoint/sft/eval_moe_metric.py \
   --gen_dir $output_path \
   --result_file $output_path/result.txt

for src in cs de et fi ru tr zh; do
    echo " $src-en  sacrebleu:" >> $output_path/result.txt
    /home/zhanglinlin/anaconda3/envs/smoe_ori/bin/python smoe/utils/eval_utils/compute_bleu.py \
       $output_path/$src-en.jsonl  en \
       >>  $output_path/result.txt
done

