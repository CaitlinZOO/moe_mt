
ROOT=/home/ubuntu          # zhanglinlin
cd ${ROOT}/pro/MoE/moe_mt
echo "进入 ${ROOT}/pro/MoE/moe_mt"

model_path=${ROOT}/outputs/moe_mt/mixtral/sft_mt/wmt18_x2-1of3/Llama-1B_8expert-MLP-MoE-Top2-Dense4_epoch0.5/

data_path=${ROOT}/zll/mt/wmt_test/wmt18_text_x-en/

output_path=${ROOT}/zll/mt/wmt_test/llama3_mixtral/output/x2-1of3_8expert-MLP-MoE-Top2-Dense4_epoch0.5 ## cs-de-ru_en
mkdir -p $output_path
touch $output_path/result.txt _mixtral.sh

CUDA_VISIBLE_DEVICES=0 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/cs-en.json --output_file $output_path/cs-en.jsonl > $output_path/cs-en.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/de-en.json --output_file $output_path/de-en.jsonl > $output_path/de-en.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/et-en.json --output_file $output_path/et-en.jsonl > $output_path/et-en.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/fi-en.json --output_file $output_path/fi-en.jsonl > $output_path/fi-en.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/ru-en.json --output_file $output_path/ru-en.jsonl > $output_path/ru-en.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/tr-en.json --output_file $output_path/tr-en.jsonl > $output_path/tr-en.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/zh-en.json --output_file $output_path/zh-en.jsonl > $output_path/zh-en.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/en-cs.json --output_file $output_path/en-cs.jsonl > $output_path/en-cs.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/en-de.json --output_file $output_path/en-de.jsonl > $output_path/en-de.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/en-et.json --output_file $output_path/en-et.jsonl > $output_path/en-et.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/en-fi.json --output_file $output_path/en-fi.jsonl > $output_path/en-fi.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/en-ru.json --output_file $output_path/en-ru.jsonl > $output_path/en-ru.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/en-tr.json --output_file $output_path/en-tr.jsonl > $output_path/en-tr.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python smoe/entrypoint/sft/test_generate_one_mixtral.py --model_path $model_path --input_file $data_path/en-zh.json --output_file $output_path/en-zh.jsonl > $output_path/en-zh.log

sleep 10min

python smoe/entrypoint/sft/eval_moe_metric.py \
   --gen_dir $output_path \
   --result_file $output_path/result.txt

for src in cs de et fi ru tr zh; do
    echo " $src-en  sacrebleu:" >> $output_path/result.txt
    python smoe/utils/eval_utils/compute_bleu.py \
       $output_path/$src-en.jsonl  en \
       >>  $output_path/result.txt
   echo " en-$src  sacrebleu:" >> $output_path/result.txt
    python smoe/utils/eval_utils/compute_bleu.py \
       $output_path/en-$src.jsonl  $src \
       >>  $output_path/result.txt
done

