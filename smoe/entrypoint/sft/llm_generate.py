import os
import argparse
import json
from tqdm import tqdm
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import string

from dataclasses import dataclass
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
# from transformers.models.llama import ll

from smoe.entrypoint.sft.train_sft_llama3_2group_mt import get_model, get_tokenizer
# from src.modeling_blsp2 import Blsp2Model
# from src.configuration_blsp2 import Blsp2Config
# from src.tokenization_qwen import QWenTokenizer
# from src.instruction_dataset import get_waveform
# from src.qwen_generation_utils import decode_tokens, get_stop_words_ids


def collate_tokens(
        values: List[List[int]],
        pad_id: int
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(torch.LongTensor(v), res[i][-len(v):])

    return res


@dataclass
class DataCollator:
    pad_id: int = 0

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        suffix_input_ids = [sample["suffix_input_ids"] for sample in samples]
        suffix_attention_mask = [sample["suffix_attention_mask"] for sample in samples]
        text = [sample["text"] for sample in samples]
        audio = None
        if "audio" in samples[0].keys():
            audio = [sample["audio"] for sample in samples]

        ## 多余
        output = None
        if "output" in samples[0].keys():
            output = [sample["output"] for sample in samples]
        early_stop = None
        if "early_stop" in samples[0].keys():
            early_stop = [sample["early_stop"] for sample in samples]
        reference = None
        if "reference" in samples[0].keys():
            reference = [sample["reference"] for sample in samples]

        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        suffix_input_ids = collate_tokens(suffix_input_ids, self.pad_id)
        suffix_attention_mask = collate_tokens(suffix_attention_mask, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            # "speech_values": speech_values,
            # "speech_attention_mask": speech_attention_mask,
            "audio": audio,
            "text": text,
            "output": output,
            "reference": reference,
            "early_stop": early_stop,
        }


# @dataclass
def generate(
        input_ids,
        attention_mask,
        suffix_input_ids,
        suffix_attention_mask,
        generation_config=None,
        stop_words_ids=None,
        qwen_model=None,
):
    inputs_embeds, input_attention_mask = [], []

    prefix_embeds = qwen_model.get_input_embeddings()(input_ids)
    inputs_embeds.append(prefix_embeds)
    input_attention_mask.append(attention_mask)

    suffix_embeds = qwen_model.get_input_embeddings()(suffix_input_ids)
    inputs_embeds.append(suffix_embeds)
    input_attention_mask.append(suffix_attention_mask)

    inputs_embeds = torch.cat(inputs_embeds, dim=1)
    input_attention_mask = torch.cat(input_attention_mask, dim=1)

    return qwen_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=input_attention_mask,
        generation_config=generation_config,
        stop_words_ids=stop_words_ids
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Path to the input file", required=True
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Path to the output file", required=True
    )
    parser.add_argument(
        "--model_type", type=str, default="mixtral2group",
        help="Path to the llm model"
    )
    parser.add_argument(
        "--llm_model", type=str, default=None,
        help="Path to the llm model", required=True
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Path to the tokenizer"
    )
    parser.add_argument(
        "--generation_config_path", type=str, default=None,
        help="Path to the generation_config"
    )
    parser.add_argument(
        "--llama_path", type=str, default="/home/zhanglinlin/models/llama/Meta-Llama-3.2-1B-Instruct",
        help="Path to the original llama path for default tokenizer and generate settings >>>"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Path to the cache "
    )
    parser.add_argument(
        "--instruction", type=str, default="",
        help="the general instruction for each example"
    )
    parser.add_argument(
        "--input_field", type=str, default="",
        help="the text field for each example"
    )
    parser.add_argument(
        "--reference_field", type=str, default="",
        help="the text field for each example"
    )
    parser.add_argument(
        "--batch_size", type=int, default=6,
        help="batch size"
    )
    ### args for generation
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="max new tokens for generation"
    )
    parser.add_argument(
        "--min_new_tokens", type=int, default=1,
        help="min new tokens for generation"
    )
    parser.add_argument(
        "--do_sample", action="store_true",
        help="whether do sample. For MT task, we will use greedy search to ensure stable output"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.5,
        help="top_p for generation"
    )
    parser.add_argument(
        "--top_k", type=int, default=0,
        help="top_k for generation"
    )
    parser.add_argument(
        "--torch_dtype", type=str, default="bfloat16",
        help="Torch dtype: `float32` or `bfloat16`"
    )
    parser.add_argument(
        "--attn_impl", type=str, default="flash_attention_2",
        help="attention implementation, choice from [eager, flash_attention_2, sdpa] (default: `flash_attention_2`)"
    )
    parser.add_argument(
        "--trust_remote_code", type=bool, default=False,
        help="trust remote code  ....   NO"
    )
    args = parser.parse_args()


    padding_side = "left"  ## for cache generation
    

    ## 1. get model and tokenizer
    if args.tokenizer_path is None:
        tokenizer_path = args.llm_model
    else:
        tokenizer_path = args.tokenizer_path
    # import pdb; pdb.set_trace() # PreTrainedTokenizer transformers.AutoTokenizer
    tokenizer = get_tokenizer(
        tokenizer_path,
        cache_dir=args.cache_dir,
        padding_side=padding_side,
        trust_remote_code=args.trust_remote_code,
    )
    model = get_model(
        args.model_type,
        args.llm_model,
        torch_dtype=args.torch_dtype,
        attn_impl=args.attn_impl,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        output_router_logits=True,
    )

    if args.generation_config_path is None:
        generation_config_path = args.llm_model
        print("从 args.llm_model 加载 Llama generation_config : ")
    else:
        generation_config_path = args.generation_config_path
        print("从 args.generation_config_path 加载 Llama generation_config : ")
    generation_config = GenerationConfig.from_pretrained(generation_config_path)
    print("  {}".format(generation_config))
    from transformers import StoppingCriteria
    import pdb; pdb.set_trace()
    stop_words_ids = get_stop_words_ids(generation_config.chat_format, tokenizer)



    with open(args.input_file, "r") as fin:
        lines = fin.readlines()
        lines = [json.loads(line.strip()) for line in lines]
    dataset = Dataset.from_list(lines)

    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def process_dataset(batch):
        def _tokenize_str(role, content):
            return tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        instruction = batch.get("instruction", args.instruction)
        if args.text_field:
            text = batch.get(args.text_field, "")
            batch["text"] = text
            instruction = instruction + text

        ### prefix
        input_ids = []
        input_ids += im_start_tokens + _tokenize_str("system", "You are a helpful assistant.") + im_end_tokens
        input_ids += nl_tokens
        input_ids += im_start_tokens + _tokenize_str("user", instruction)
        batch["input_ids"] = input_ids
        batch["attention_mask"] = [1] * len(batch["input_ids"])
        ### audio
        if args.audio_field:
            batch["audio"] = batch.get(args.audio_field, "")  ## 原来默认是 None
        else:
            batch["audio"] = None
        ### suffix
        suffix_input_ids = im_end_tokens + nl_tokens + im_start_tokens + \
                           tokenizer.encode("assistant")  # \n is removed and used as bos_token
        batch["suffix_input_ids"] = suffix_input_ids
        batch["suffix_attention_mask"] = [1] * len(batch["suffix_input_ids"])
        ### reference
        if args.reference_field:
            batch["reference"] = batch.get(args.reference_field, "")  ## 原来默认是 None
        else:
            batch["reference"] = None
        return batch

    dataset = dataset.map(process_dataset)
    model = QWenLMHeadModel.from_pretrained(args.llm_model, torch_dtype=torch.float16)
    model = model.half()  ### on A100, Qwen will automatically converting to bf16

    data_collator = DataCollator(generation_config.pad_token_id)
    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size
    )

    generation_config.update(
        **{
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_beams": 1,
            "num_return_sequences": 1,
            "bos_token_id": nl_tokens[0],  ### need bos_token_id when input_ids is not provided
        }
    )

    model = model.cuda()
    model.eval()
    with open(args.output_file, "w") as fout:
        for batch in tqdm(dataloader):
            outputs = generate(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
                generation_config=generation_config,
                stop_words_ids=stop_words_ids,
                qwen_model=model,
            )
            # speech_values=batch["speech_values"].cuda() if batch["speech_values"] is not None else None,
            # speech_attention_mask=batch["speech_attention_mask"].cuda() if batch["speech_attention_mask"] is not None else None,
            output_text = [
                decode_tokens(
                    output,
                    tokenizer,
                    raw_text_len=0,
                    context_length=0,
                    chat_format=generation_config.chat_format,
                    verbose=False,
                    errors='replace'
                )
                for output in outputs
            ]
            for audio, text, output, reference, early_stop, response in zip(batch["audio"], batch["text"],
                                                                            batch["output"], batch["reference"],
                                                                            batch["early_stop"], output_text):
                response_new1 = re.sub("^\\\"|\\\"$", "", response)
                split_s = response_new1.split("\\\"")
                if len(split_s) > 1:
                    split_2 = split_s[1]
                    if len(split_2) > 2:
                        response_new1 = split_2
                response_new2 = response_new1.replace("\\\"", " ")
                if output is not None and text is not None and text != "":
                    json_string = json.dumps(
                        {
                            "audio": audio,
                            "text": text,
                            "output": output,
                            "early_stop": early_stop,
                            "response": response_new2,
                            "reference": reference,
                        },
                        ensure_ascii=False,
                    )
                else:
                    json_string = json.dumps(
                        {
                            "audio": audio,
                            "text": text,
                            "response": response_new2,
                            "reference": reference,
                        },
                        ensure_ascii=False,
                    )
                print(json_string, file=fout, flush=True)  ## 为什么 flush 等于True


if __name__ == "__main__":
    main()
