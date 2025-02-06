
import json
import os
from tqdm import tqdm 
import torch
# import logging
import tiktoken

from transformers import AutoModelForCausalLM, AutoTokenizer
from smoe.models.mixtral_2group import Mixtral2GroupConfig, Mixtral2GroupForCausalLM
from smoe.utils.conversation import Llama3ConversationTemplate


global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
max_memory = {k: '32GB' for k in global_devices}

# print(global_devices)
# exit()
model_name_or_path = "/ssd3/data/acl25/lora/x-en/lama3_1b_merge"

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map='balanced',
    max_memory=max_memory
)
config = Mixtral2GroupConfig.from_pretrained(model_name_or_path)
config._attn_implementation = "flash_attention_2"
config.output_router_logits = True
model = Mixtral2GroupForCausalLM.from_pretrained(
     model_name_or_path,
    config=config,
    torch_dtype="auto",
    device_map='balanced',
    max_memory=max_memory
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tt = tokenizer.encode(text="\n\n")
if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id

encoding = tiktoken.get_encoding("cl100k_base")
def num_tokens_from_string(text):
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(text))
    return num_tokens


# cs-en.json  de-en.json  et-en.json  fi-en.json  ru-en.json  tr-en.json  zh-en.json

langs = [
    'cs',
    # 'de',
    # 'et',
    # 'fi',
    # 'ru',
    # 'tr',
    # 'zh'
]
root = '/home/zhanglinlin/zll/mt/wmt_test/wmt18_text_x-en'
def main():

    for lang in langs:
        file_path = root + lang +'-en.json' 
        output_file = f'/home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0/{lang}-en.jsonl'
        # 确保文件存在
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过...")
            continue
        with open(file_path, "r", encoding="utf-8") as f, open(output_file, 'w', encoding='utf-8') as out_file:
                data = json.load(f)  # 读取 JSON 文件（假设是 JSON 数组）
                # print(data)
                for i, item in  enumerate(data):
                    print(i ,item)
                    print('====')
                    # exit()



                    # prompt = item['instruction']+ item['input']+ ".\n"
                    llama_template = Llama3ConversationTemplate()
                    begin_text_tokens = [128000]  ## "<|begin_of_text|>
                    end_text_tokens = [128001]  ## "<|end_of_text|>"
                    im_end_tokens = [128009]  ##  self.eot: str = "<|eot_id|>"
                    nl_tokens = tokenizer.encode(text="\n\n")  ## llama
                    prompt = llama_template.get_context_str(role="system", context="You are a helpful assistant.", add_eos=False)
                    prompt += llama_template.get_context_str(role="user", context="", add_eos=False)
                    prompt += llama_template.get_context_str(context=item['instruction'], add_eos=False)
                    prompt += nl_tokens + llama_template.get_context_str(context=item['input'], add_eos=False)
                    prompt += llama_template.get_context_str(role="assistant", context="", add_eos=False) + nl_tokens
                    # print(prompt)


                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                    # Generate
                    generate_ids = model.generate(
                        inputs.input_ids, 
                        max_new_tokens=num_tokens_from_string(item['input'])*2,
                        temperature=0.1,
                        do_sample=True)
                    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    print(response+"\n======")
                    response = response[len(prompt):].strip().split('\n')[0]
                    item['pred_text'] = response
                    item['id'] = i 
                    res = json.dumps(item, ensure_ascii=False)
                    out_file.write(f"{res}\n")
                    # i += 1
                    out_file.flush()

                    # print(res)
                    # exit()

if __name__ == '__main__':
    main()
