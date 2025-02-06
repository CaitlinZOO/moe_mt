
import json
import os
from tqdm import tqdm 
import torch
# import logging
import tiktoken

from transformers import AutoModelForCausalLM, AutoTokenizer
from smoe.models.mixtral_2group import Mixtral2GroupConfig, Mixtral2GroupForCausalLM
from smoe.utils.conversation import Llama3ConversationTemplate
from smoe.utils.datasets.text_instruction_dataset import collate_tokens


global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
max_memory = {k: '20GB' for k in global_devices}

# print(global_devices)
# exit()
model_name_or_path = "/home/zhanglinlin/outputs/moe_mt/mixtral_2group/sft_mt/Llama-1B_2group_4experts_top1_fft-wm18-1of3"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     torch_dtype="auto",
#     device_map='balanced',
#     max_memory=max_memory,
# )
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
model.set_groups_used([0, 1])  ## 运行mt任务

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

# tokenizer = LlamaTokenizer
llama_template = Llama3ConversationTemplate()
def _tokenize_str(role="", content=""):
        ### tokenizer.encode编码成ids，设不设置 allowed_special=set() 结果一样，不添加特殊符号
        tokens = []
        con_str = llama_template.get_context_str(
            role=role, context=content, add_eos=False
        )
        # tokens += LlamaTokenizer.encode(con_str, add_special_tokens=False)
        # tokens += tokenizer.encode(text=con_str)
        tokens += tokenizer.encode(text=con_str, add_special_tokens=False)
        # import pdb;pdb.set_trace()
        return tokens

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
root = '/home/zhanglinlin/zll/mt/wmt_test/wmt18_text_x-en/'
def main():

    for lang in langs:
        file_path = root + lang +'-en.json' 
        output_file = f'/home/zhanglinlin/zll/mt/wmt_test/llama3_mixtral2group/output/1of3_0-t/{lang}-en.jsonl'
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
                    begin_text_tokens = [128000]  ## "<|begin_of_text|>"
                    end_text_tokens = [128001]  ## "<|end_of_text|>"
                    im_end_tokens = [128009]  ##  self.eot: str = "<|eot_id|>"
                    nl_tokens = tokenizer.encode(text="\n\n", add_special_tokens=False)  ## llama

                    prompt = "<|begin_of_text|>" + llama_template.get_context_str(role="system", context="You are a helpful assistant.", add_eos=False)
                    prompt += llama_template.get_context_str(role="user", context="", add_eos=False)
                    prompt += "\n\n" + llama_template.get_context_str(context=item['instruction'], add_eos=False)
                    prompt += "\n\n" + "<|begin_of_text|>" + item['input'] + "<|end_of_text|>" + "<|eot_id|>"
                    prompt += llama_template.get_context_str(role="assistant", context="", add_eos=False) + "\n\n"
                    # print(prompt)
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


                    input_ids = begin_text_tokens + _tokenize_str(role="system", content="You are a helpful assistant.")
                    input_ids += _tokenize_str(role="user") 
                    input_ids += nl_tokens + _tokenize_str(content=item['instruction'])
                    input_ids += nl_tokens + begin_text_tokens + tokenizer.encode(text=item['input'], add_special_tokens=False) + end_text_tokens + im_end_tokens
                    input_ids += _tokenize_str(role="assistant") + nl_tokens
                    input_ids = collate_tokens([input_ids], tokenizer.pad_token_id, padding_side="left").to(model.device)
                    prompt_0 = tokenizer.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

                    # Generate
                    # generate_ids = model.generate(
                    #     inputs.input_ids, 
                    #     max_new_tokens=num_tokens_from_string(item['input'])*2,
                    #     temperature=0.1,
                    #     do_sample=True)
                    generate_ids_0 = model.generate(
                        input_ids, 
                        max_new_tokens=num_tokens_from_string(item['input'])*2,
                        temperature=0.1,
                        do_sample=True,
                        tokenizer=tokenizer,
                        )
                    # response = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                    response_0 = tokenizer.batch_decode(generate_ids_0, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                    print(response_0+"\n======")
                    import pdb; pdb.set_trace()
                    # response = response[len(prompt):].strip().split('\n')[0]
                    # response_0 = response_0[len(prompt_0):].removeprefix('<|begin_of_text|>').split("<|eot_id|>")[0]
                    response_0 = response_0[len(prompt_0):].removeprefix('<|begin_of_text|>').split("<|eot_id|>")[0].split("<|end_of_text|>")[0]
                    item['pred_text'] = response_0
                    item['id'] = i 
                    res = json.dumps(item, ensure_ascii=False)
                    out_file.write(f"{res}\n")
                    # i += 1
                    out_file.flush()

                    # print(res)
                    # exit()

if __name__ == '__main__':
    main()
