import copy
import csv
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import datasets
import fire
import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

# from transformers.models.llama import tokenization_llama
from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer

# from src.configuration_qwen import QWenConfig
# from src.modeling_qwen import QWenLMHeadModel
# from src.tokenization_qwen import QWenTokenizer
import smoe.models.mixtral_2group.modeling_mixtral2group as ModelingMixtral2groupResidual
from smoe.models.mixtral_2group import Mixtral2GroupConfig, Mixtral2GroupForCausalLM

# from src.qwen_generation_utils import decode_tokens, get_stop_words_ids
from smoe.utils.conversation import Llama3ConversationTemplate

logger = logging.getLogger(__name__)


Text_Format = (
    # "###[Human]:{instruction}\n\n\n"
    # "###[Assistant]:"
    "<|start_header_id|> human <|end_header_id|>\n\n{message}<|eot_id|>"
    "<|start_header_id|> assistant <|end_header_id|>"
)

instruction_dict = {
    # "en": {
    "translate": "Please translate the following English text into {target} text:",
    # }
}


def process_dataset(
    batch,
    tokenizer,
    _tokenize_str,
    instruction="",
    instruction_field="",
    input_field="input",
    output_field="output",
    max_length=384,
    check_audio=False,
):
    to_keep = True
    if not input_field:
        raise ValueError(f" input_field not set for processing batch: {batch}")
    if not output_field:
        raise ValueError(f"output_field not set for processing batch: {batch}")
    if instruction_field:
        instruction = batch[instruction_field]  ## batch["instruction"]

    begin_text_tokens = [128000]  ## "<|begin_of_text|>
    end_text_tokens = [128001]  ## "<|end_of_text|>"
    im_end_tokens = [128009]  ##  self.eot: str = "<|eot_id|>"
    nl_tokens = tokenizer.encode(text="\n\n", add_special_tokens=False)  ## llama

    start_ids = []  ##  self.eot: str = "<|eot_id|>"
    start_ids += begin_text_tokens + _tokenize_str(role="system", content="You are a helpful assistant.")
    # start_ids += nl_tokens
    start_ids += _tokenize_str(role="user")
    start_mask = [1] * len(start_ids)
    start_labels = [-100] * len(start_ids)

    instruction_ids, instruction_mask, instruction_labels = [], [], []
    if instruction:
        instruction_ids = nl_tokens + _tokenize_str(content=instruction)
        instruction_mask = [1] * len(instruction_ids)
        instruction_labels = [-100] * len(instruction_ids)

    input_ids, input_mask, input_labels = [], [], []
    if input_field:
        ## nl_tokens + _tokenize_str(content=batch[input_field])
        input_ids = nl_tokens + begin_text_tokens + tokenizer.encode(text=batch[input_field], add_special_tokens=False) + end_text_tokens + im_end_tokens
        input_mask = [1] * len(input_ids)
        input_labels = [-100] * len(input_ids)  ##   input_ids
        if len(batch[input_field]) < 1 or len(input_ids) < 2:
            to_keep = False

    suffix_ids, suffix_mask, suffix_labels = [], [], []
    new_ids = _tokenize_str(role="assistant")
    suffix_ids += new_ids
    suffix_mask += [1] * len(new_ids)
    suffix_labels += [-100] * len(new_ids)

    if output_field:
        # nl_tokens + _tokenize_str(content=batch[output_field])
        new_ids = nl_tokens + begin_text_tokens + tokenizer.encode(text=batch[output_field], add_special_tokens=False) + end_text_tokens + im_end_tokens
        if len(batch[output_field]) < 1:
            to_keep = False
    else:
        new_ids = im_end_tokens
    suffix_ids += new_ids
    suffix_mask += [1] * len(new_ids)
    suffix_labels += new_ids

    if (
        len(start_ids) + len(instruction_ids) + len(input_ids) + len(suffix_ids)
    ) > max_length:
        to_keep = False

    ### 只做 lm 任务 不加instruct
    if (
        instruction is None or instruction == [] or instruction == ""
    ):  # and input_field == output_field
        ###  qwen_tokenizer.eod_id 151643   qwen_tokenizer.im_start_id 151644
        ###
        # import pdb; pdb.set_trace()
        # begin_text_tokens + nl_tokens
        start_ids = begin_text_tokens + nl_tokens
        start_mask = [1] * len(start_ids)
        start_labels = [-100] * len(start_ids)
        instruction_ids, instruction_mask, instruction_labels = [], [], []
        if len(batch[input_field]) < 5:
            to_keep = False
        input_str = batch[input_field] + "<|end_of_text|>"
        # _tokenize_str(content=input_str)
        input_ids = tokenizer.encode(text=input_str, add_special_tokens=False)
        input_mask = [1] * len(input_ids)
        input_labels = copy.deepcopy(input_ids)  ## [-100] * len(input_ids)
        input_labels[:4] = [-100] * 4
        input_labels[-2:] = [-100] * 2
        suffix_ids = im_end_tokens
        suffix_mask = [1] * len(suffix_ids)
        suffix_labels = [-100] * len(suffix_ids)  ##suffix_ids
        if (len(start_ids) + len(input_ids) + len(suffix_ids)) > max_length:
            to_keep = False
    if np.random.random() < 0.0000001:
        print("instruction : {} \n {}".format(len(instruction.split()), instruction))
        input_text = batch[input_field]
        output_text = batch[output_field]
        print("input_text : {} \n {}".format(len(input_text.split()), input_text))
        print("output_text : {} \n {}".format(len(output_text.split()), output_text))
        print("start_ids : {} \n {}".format(len(start_ids), start_ids))
        print("instruction_ids : {} \n {}".format(len(instruction_ids), instruction_ids))
        print("input_ids : {} \n {}".format(len(input_ids), input_ids))
        print("suffix_ids : {} \n {}\n".format(len(suffix_ids), suffix_ids))
        all_input = start_ids + instruction_ids + input_ids + suffix_ids
        print(tokenizer.batch_decode([all_input], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])

    # batch["start_ids"] = start_ids
    # batch["start_mask"] = start_mask
    # batch["start_labels"] = start_labels
    # batch["instruction_ids"] = instruction_ids
    # batch["instruction_mask"] = instruction_mask
    # batch["instruction_labels"] = instruction_labels
    # batch["input_ids"] = input_ids
    # batch["input_mask"] = input_mask
    # batch["input_labels"] = input_labels
    # batch["suffix_ids"] = suffix_ids
    # batch["suffix_mask"] = suffix_mask
    # batch["suffix_labels"] = suffix_labels

    batch["input_ids"] = start_ids + instruction_ids + input_ids + suffix_ids
    batch["attention_mask"] = start_mask + instruction_mask + input_mask + suffix_mask
    batch["labels"] = start_labels + instruction_labels + input_labels + suffix_labels

    batch["to_keep"] = to_keep

    return batch


def load_text_instruction_dataset(
    # dataset_save_dir="",
    # manifest_dir="",
    manifest_files="",
    tokenizer=None,
    instruction="",
    instruction_field="",
    input_field="",
    output_field="",
    max_length=384,
    num_proc=8,
):
    # if os.path.exists(os.path.join(dataset_save_dir, f"processed_{manifest_files}")):
    #     logger.warning("load processed dataset")
    #     dataset = datasets.load_from_disk(os.path.join(dataset_save_dir, f"processed_{manifest_files}"))
    #     return dataset

    # logger.warning(f"load dataset from scratch from {dataset_save_dir}/{manifest_files}")

    manifest_files_list = manifest_files.split(",")
    # raw_dataset = datasets.load_dataset(
    #         manifest_dir, data_files=manifest_files_list, split="train", streaming=False
    #     )
    if manifest_files_list[0].endswith("jsonl") or manifest_files_list[0].endswith(
        "json"
    ):
        print("文件格式 是 .json ")
        raw_dataset = datasets.load_dataset(
            "json", data_files=manifest_files_list, split="train", streaming=False
        )
    elif manifest_files_list[0].endswith("tsv"):
        print("文件格式 是 .tsv ")
        text_datasets = []
        for text_file in manifest_files_list:
            tsv_path = Path(text_file)  # / f"{text_file}"
            if not tsv_path.is_file():
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            tsv_path = str(tsv_path)
            text_dataset = Dataset.from_csv(
                tsv_path,
                split="train",
                streaming=False,
                delimiter="\t",
                lineterminator="\n",
                quotechar=None,
                doublequote=False,
                quoting=csv.QUOTE_NONE,
            )  ## , sep="\t"
            text_datasets.append(text_dataset)
        raw_dataset = datasets.concatenate_datasets(text_datasets)
    else:
        print("文件格式不对，不是 .json 或者 .tsv ")

    # tokenizer = LlamaTokenizer
    llama_template = Llama3ConversationTemplate()

    def _tokenize_str(role="", content=""):
        ### tokenizer.encode编码成ids，设不设置 allowed_special=set() 结果一样，不添加特殊符号
        tokens = []
        con_str = llama_template.get_context_str(
            role=role, context=content, add_eos=False
        )
        # tokens += LlamaTokenizer.encode(con_str, add_special_tokens=False)
        tokens += tokenizer.encode(text=con_str, add_special_tokens=False) ## 修改，不添加 [128000]  ## "<|begin_of_text|>
        return tokens

    dataset = raw_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "_tokenize_str": _tokenize_str,
            "instruction": instruction,
            "instruction_field": instruction_field,
            "input_field": input_field,
            "output_field": output_field,
            "max_length": max_length,
        },
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
        num_proc=num_proc,
    )

    def to_keep(flag):
        return flag

    dataset = dataset.filter(to_keep, input_columns=["to_keep"])
    #
    # dataset.save_to_disk(os.path.join(dataset_save_dir, f"processed_{manifest_files}"))
    return dataset


def load_text_instruction_datasets(data_args, tokenizer=None, num_proc=1, do_eval=False):
    dataset = None
    dataset_save_dir = data_args.dataset_save_dir if os.path.exists(data_args.dataset_save_dir) else None
    if dataset_save_dir is not None and do_eval:
        dataset_save_dir = data_args.dataset_save_dir + "_eval"
        print("dataset_save_dir :{} ".format(dataset_save_dir))
    if os.path.exists(dataset_save_dir) and os.listdir(dataset_save_dir):
        try:
            logger.warning(f"loading processed dataset from {dataset_save_dir}")
            dataset = datasets.load_from_disk(dataset_save_dir)
            return dataset
        except:
            logger.warning(f" load from {dataset_save_dir},   fail !! ")

    manifest_keys = ["manifest_files", "instructions", "instruction_fields", "input_fields", "output_fields"]
    if do_eval:
        manifest_keys = ["eval_manifest_files", "eval_instructions", "eval_instruction_fields", "eval_input_fields", "eval_output_fields"]
    
    if dataset is not None:
        num_datasets = len(dataset)
    else:
        manifest_values = [
            (getattr(data_args, key)).split("|") for key in manifest_keys
        ]
        num_datasets = len(manifest_values[0])
        print("num_datasets : {}".format(num_datasets))
        if num_datasets == 0:
            raise ValueError("no datasets specified")
        for i, key in enumerate(manifest_keys):
            if len(manifest_values[i]) != num_datasets:
                raise ValueError(f"unexpected number of {key} in {data_args}")
        all_datasets = [
            load_text_instruction_dataset(
                manifest_files=manifest_values[0][i],
                instruction=manifest_values[1][i],
                instruction_field=manifest_values[2][i],
                input_field=manifest_values[3][i],
                output_field=manifest_values[4][i],
                tokenizer=tokenizer,
                num_proc=num_proc,
            )
            for i in range(num_datasets)
        ]
    
    if len(all_datasets) == 1:
        dataset = all_datasets[0]
    else:
        sample_probs = [float(prob) for prob in data_args.sample_probs.split("|")]
        if len(sample_probs) != num_datasets:
            raise ValueError(f"unexpected number of probabilities in {data_args}")
        if sum(sample_probs) != 1:
            dataset = datasets.concatenate_datasets(all_datasets)
        else:  ## 概率和 为 1
            dataset = datasets.interleave_datasets(
                all_datasets,
                stopping_strategy=data_args.interleave_stopping_strategy,
                probabilities=sample_probs,
            )

    print("dataset_save_dir : {}".format(dataset_save_dir))
    if dataset_save_dir and (
        not dist.is_initialized() or dist.get_rank() == 0
    ):
        # if not os.path.exists(dataset_save_dir):
        #     os.mkdir(dataset_save_dir)
        dataset.save_to_disk(dataset_save_dir)

    return dataset


def collate_tokens(values: List[List[int]], pad_id: int, padding_side: str = "right"):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        if padding_side == "right":
            copy_tensor(torch.LongTensor(v), res[i][: len(v)])
        else:
            copy_tensor(torch.LongTensor(v), res[i][-len(v):])

    return res

@dataclass
class TextInstructionDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """

    # Simple for llama3, we directly use '<|eot_id|>' (128009) for pad token. You should change for other models.
    pad_id: int = 128009
    padding_side: str = "right"  ## default is "right", but for using the cache to text generation must be setted "left"

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        # print("input_ids : {} \n {}".format(len(input_ids), input_ids))
        # print("attention_mask : {} \n {}".format(len(attention_mask), attention_mask))
        # print("labels : {} \n {}".format(len(labels), labels))

        input_ids = collate_tokens(input_ids, self.pad_id, padding_side= self.padding_side)
        attention_mask = collate_tokens(attention_mask, 0, padding_side= self.padding_side)  ## 1, 0
        labels = collate_tokens(labels, -100, padding_side= self.padding_side)

        # print("input_ids : {} \n {}".format(input_ids.shape, input_ids))
        # print("attention_mask : {} \n {}".format(attention_mask.shape, attention_mask))
        # print("labels : {} \n {}".format(labels.shape, labels))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class TextInstructionDataCollatorALL:
    """
    Data collator that will dynamically pad the inputs received.
    """

    # Simple for llama3, we directly use '<|eot_id|>' (128009) for pad token. You should change for other models.
    pad_id: int = 128009

    def __call__(self, samples: List[Dict]):
        start_ids = [sample["start_ids"] for sample in samples]
        start_mask = [sample["start_mask"] for sample in samples]
        start_labels = [sample["start_labels"] for sample in samples]
        instruction_ids = [sample["instruction_ids"] for sample in samples]
        instruction_mask = [sample["instruction_mask"] for sample in samples]
        instruction_labels = [sample["instruction_labels"] for sample in samples]
        input_ids = [sample["input_ids"] for sample in samples]
        input_mask = [sample["input_mask"] for sample in samples]
        input_labels = [sample["input_labels"] for sample in samples]
        suffix_ids = [sample["suffix_ids"] for sample in samples]
        suffix_mask = [sample["suffix_mask"] for sample in samples]
        suffix_labels = [sample["suffix_labels"] for sample in samples]

        start_ids = collate_tokens(start_ids, self.pad_id)
        start_mask = collate_tokens(start_mask, 0)
        start_labels = collate_tokens(start_labels, -100)
        # if instruction_ids is not None:
        instruction_ids = collate_tokens(instruction_ids, self.pad_id)
        instruction_mask = collate_tokens(instruction_mask, 0)
        instruction_labels = collate_tokens(instruction_labels, -100)
        input_ids = collate_tokens(input_ids, self.pad_id)
        input_mask = collate_tokens(input_mask, 0)
        input_labels = collate_tokens(input_labels, -100)
        suffix_ids = collate_tokens(suffix_ids, self.pad_id)
        suffix_mask = collate_tokens(suffix_mask, 0)
        suffix_labels = collate_tokens(suffix_labels, -100)

        return {
            "start_ids": start_ids,
            "start_mask": start_mask,
            "start_labels": start_labels,
            "instruction_ids": instruction_ids,
            "instruction_mask": instruction_mask,
            "instruction_labels": instruction_labels,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "input_labels": input_labels,
            "suffix_ids": suffix_ids,
            "suffix_mask": suffix_mask,
            "suffix_labels": suffix_labels,
        }


def offline_process(
    # dataroot="",
    manifest_files="/home/zhanglinlin/zll/mt/wmt_test/wmt18_text_x-en/de-en.json",
    tokenizer="",
    instruction="",
    instruction_field="instruction",
    input_field="input",
    output_field="output",
    max_length=384,
    num_proc=8,
):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="/home/zhanglinlin/outputs/moe_mt/converted_models/Llama3.2-1B-2group-4-4expert-MLP-MoE-Top1-Scale4.0-Insert4_use-fft",
        cache_dir=None,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    dataset = load_text_instruction_dataset(
        # dataroot,
        manifest_files,
        tokenizer=tokenizer,
        instruction=instruction,
        instruction_field=instruction_field,
        input_field=input_field,
        output_field=output_field,
        max_length=max_length,
        num_proc=num_proc,
    )
    print(len)
    print(dataset)
    for key in dataset[0].keys():
        print(key, dataset[0][key])


if __name__ == "__main__":
    fire.Fire(
        {
            "offline": offline_process,
        }
    )
    # offline_process(
    #     manifest_files="/home/zhanglinlin/zll/mt/wmt_test/wmt18_text_x-en/de-en.json",
    # tokenizer="",
    # instruction="",
    # instruction_field="instruction",
    # input_field="input",
    # output_field="output",
    # max_length=384,
    # num_proc=1,
    # )
