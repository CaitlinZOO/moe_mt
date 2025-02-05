import logging
import math
import os
import pathlib
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import datasets
import numpy as np
import torch
import transformers
from loguru import logger
from torch.utils.data import Dataset
from transformers import GenerationConfig, PreTrainedTokenizer, Trainer, set_seed
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.trainer_pt_utils import LabelSmoother

import smoe.models.mixtral_2group.modeling_mixtral2group as ModelingMixtral2groupResidual
from smoe.models.mixtral_2group import Mixtral2GroupConfig, Mixtral2GroupForCausalLM
from smoe.utils.conversation import Llama3ConversationTemplate

from sklearn.metrics import accuracy_score  ## , precision_recall_fscore_support
# from datasets import load_metrix  ## DatasetDict, load_dataset, Dataset
from smoe.utils.datasets.text_instruction_dataset import (
    TextInstructionDataCollator,
    load_text_instruction_datasets,
)
from smoe.utils.io import get_pathname_from_name_or_path, load_json, load_jsonlines

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    model_type: str = field(
        default="auto",
        metadata={"help": "Model type: `moe` or `mixtral` or 'model_type' or `auto`"},
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Torch dtype: `float32` or `bfloat16`"},
    )
    additional_config: str = field(
        default=None,
        metadata={"help": "Additional config file (in json) to load"},
    )
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={
            "help": "attention implementation, choice from [eager, flash_attention_2, sdpa] (default: `flash_attention_2`)"
        },
    )

    def __post_init__(self):
        if hasattr(torch, self.torch_dtype):
            self.torch_dtype = getattr(torch, self.torch_dtype)
        if self.additional_config is not None:
            if not pathlib.Path(self.additional_config).exists():
                raise ValueError(
                    f"Additional config file {self.additional_config} not found"
                )
            self.additional_config = load_json(self.additional_config)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # eval_data_dir: str = field(
    #     default=None, metadata={"help": "Path to the evaluation data folder."}
    # )
    # dataset_dir_or_path: str = field(
    #     default="data/merged",
    #     metadata={"help": "Path to dataset directory or a single jsonl file"},
    # )
    dataset_save_dir: str = field(
        default="",
        metadata={
            "help": "directories (separated by '|') to load and save processed datasets, other data "
            "arguments ignored if set"
        },
    )
    manifest_files: str = field(
        default="",
        metadata={
            "help": "manifest files (separated by '|' between datasets and then ',' between files) "
            "of the training manifest files"
        },
    )
    instructions: str = field(
        default="",
        metadata={
            "help": "instruction_fields (separated by '|'( to read from manifest_files"
        },
    )
    instruction_fields: str = field(
        default="", metadata={"help": "instruction_fields (separated by '|'( to read from manifest_files"}
    )
    input_fields: str = field(
        default="",
        metadata={
            "help": "input_fields (separated by '|') to read from manifest_files"
        },
    )
    output_fields: str = field(
        default="",
        metadata={
            "help": "output_fields (separated by '|') to read from manifest_files"
        },
    )
    sample_probs: str = field(
        default="",
        metadata={
            "help": "sample_probs (separated by '|') for each dataset (needed for more than one "
            "dataset)"
        },
    )
    eval_manifest_files: str = field(
        default=None,
        metadata={
            "help": "manifest files (separated by '|' between datasets and then ',' between files) "
            "of the training manifest files"
        },
    )
    eval_instructions: str = field(
        default="",
        metadata={
            "help": "instruction_fields (separated by '|'( to read from manifest_files"
        },
    )
    eval_instruction_fields: str = field(
        default="", metadata={"help": "instruction_fields (separated by '|'( to read from manifest_files"}
    )
    eval_input_fields: str = field(
        default="",
        metadata={
            "help": "input_fields (separated by '|') to read from manifest_files"
        },
    )
    eval_output_fields: str = field(
        default="",
        metadata={
            "help": "output_fields (separated by '|') to read from manifest_files"
        },
    )
    eval_sample_probs: str = field(
        default="",
        metadata={
            "help": "sample_probs (separated by '|') for each dataset (needed for more than one "
            "dataset)"
        },
    )

    max_length: int = field(
        default=1024,
        metadata={
            "help": "samples that have more text tokens than this limit are removed"
        },
    )
    # dataset_save_dir: str = field(
    #     default="", metadata={"help": "save the resulting dataset for future use"}
    # )
    interleave_stopping_strategy: str = field(
        default="first_exhausted",
        metadata={
            "help": "choose from 'first_exhausted' (default) and 'all_exhausted'"
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    freeze_gate: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the gate during training."},
    )
    output_router_logits: bool = field(
        default=True,
        metadata={"help": "Whether to output router logits."},
    )
    use_layer_wise_balance: bool = field(
        default=True,
        metadata={"help": "Whether to use router BL loss."},
    )
    save_final_ckpt: bool = field(
        default=True,
        metadata={"help": "Whether to save final checkpoint."},
    )
    # max_grad_norm: float = field(
    #     default=1.0,
    #     metadata={"help": "Max gradient norm."},
    # )
    save_only_model: bool = field(
        default=False,
        metadata={"help": "Whether to save optimizer."},
    )


def trainer_save_model_safe(trainer):
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def get_tokenizer(
    model_name_or_path,
    cache_dir: str = None,
    model_max_length: int = 2048,
    padding_side: str = "right",
    use_fast: bool = False,
    trust_remote_code: bool = False,
):
    # import pdb; pdb.set_trace() # PreTrainedTokenizer transformers.AutoTokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
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
    logger.info(f"tokenizer ready, pad_token: {tokenizer.pad_token}")
    print(f"tokenizer ready, pad_token: {tokenizer.pad_token}")
    return tokenizer


def get_model(
    model_type: str,
    model_name_or_path: str,
    torch_dtype: str = "auto",
    model_max_length: int = 2048,
    attn_impl: str = "flash_attention_2",
    cache_dir: str = None,
    trust_remote_code: bool = False,
    output_router_logits: bool = True,
    use_layer_wise_balance: bool = True,
    additional_config: dict = None,
):
    logger.info(f"Model type: {model_type}")
    if model_type == "auto":
        ConfigClass = transformers.AutoConfig
        ModelClass = transformers.AutoModelForCausalLM
    elif model_type == "mixtral2group":
        ConfigClass = Mixtral2GroupConfig
        ModelClass = Mixtral2GroupForCausalLM
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Set RoPE scaling factor
    config = ConfigClass.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    config._attn_implementation = attn_impl
    config.output_router_logits = output_router_logits
    config.use_layer_wise_balance = use_layer_wise_balance
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    print(" max_position_embeddings : {}    --".format(orig_ctx_len))
    if orig_ctx_len and model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    if additional_config is not None:
        config.update(additional_config)
    logger.info("Config ready")

    # Load model and tokenizer
    model = ModelClass.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    # model.to('cuda')
    # model = ModelClass(config)
    logger.info(
        "从 model_args.model_name_or_path 加载 Llama 模型权重。\n {}:{}".format(
            model_type, model_name_or_path
        )
    )
    logger.info("model ready")

    return model


def get_model_and_tokenizer(
    model_type: str,
    model_name_or_path: str,
    tokenizer_path: str = None,
    torch_dtype: str = "auto",
    model_max_length: int = 2048,
    attn_impl: str = "flash_attention_2",
    cache_dir: str = None,
    trust_remote_code: bool = False,
    output_router_logits: bool = True,
    use_layer_wise_balance: bool = True,
    padding_side: str = "right",
    additional_config: dict = None,
    use_fast: bool = False,
) -> tuple:
    if tokenizer_path is None:
        tokenizer_path = model_name_or_path
    tokenizer = get_tokenizer(
        tokenizer_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    model = get_model(
        model_type,
        model_name_or_path,
        torch_dtype=torch_dtype,
        model_max_length=model_max_length,
        attn_impl=attn_impl,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        output_router_logits=output_router_logits,
        use_layer_wise_balance=use_layer_wise_balance,
        additional_config=additional_config,
    )

    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    logger.info(
        "从 model_args.model_name_or_path 加载 Llama generation_config : \n {}\n".format(
            generation_config
        )
    )

    return model, tokenizer, generation_config


class llamaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # """
        # How the loss is computed by Trainer. By default, all models return the loss in the first element.

        # Subclass and override for custom behavior.
        # """
        for name, param in model.named_parameters():
            # logger.info(name, param.shape, param.numel())
            # print("{} : {} : {}".format(name, param.shape, param.requires_grad))
            param.requires_grad = True

        if return_outputs:
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=return_outputs
            )
        else:
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        for name, param in model.named_parameters():
            if "block_mlp_moe" in name and "groups.1" in name:  ##  只 ft 專家
                param.requires_grad = True
                # print("block_mlp_moe  groups.1 {} : {} : {}".format(name, param.shape, param.requires_grad))
            else:
                param.requires_grad = False

        return (loss, outputs) if return_outputs else loss


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    # donot report to tensorboard, may speed up training
    # training_args: TrainingArguments
    # if "tensorboard" not in training_args.report_to:
    #     training_args.report_to.append("tensorboard")
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    # # 2. Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    model, tokenizer, generation_config = get_model_and_tokenizer(
        model_args.model_type,
        model_args.model_name_or_path,
        tokenizer_path=model_args.tokenizer_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side=model_args.padding_side,
        torch_dtype=model_args.torch_dtype,
        additional_config=model_args.additional_config,
        output_router_logits=training_args.output_router_logits,
        use_layer_wise_balance=training_args.use_layer_wise_balance,
        attn_impl=model_args.attn_impl,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    tot_params = 0
    for name, param in model.named_parameters():
        # logger.info(name, param.shape, param.numel())
        # print("{} : {} : {}".format(name, param.shape, param.requires_grad))
        if "block_mlp_moe" in name and "groups.1" in name:  ##  只 ft 專家
            param.requires_grad = True
            # print("block_mlp_moe   {} : {} : {}".format(name, param.shape, param.requires_grad))
        else:
            param.requires_grad = False
        tot_params += param.numel()
    logger.info(f"  groups.1     will trained!   --")
    logger.info(f"Total model params: {tot_params}")

    # if training_args.freeze_gate:
    #     for name, param in model.named_parameters():
    #         if "block_mlp_moe" in name and ".gate." in name:
    #             param.requires_grad = False

    train_dataset = None
    ### 5. Load dataset
    # Simple for llama3, we directly use '<|eot_id|>' (128009) for pad token. You should change for other models.
    train_dataset = load_text_instruction_datasets(data_args, tokenizer=tokenizer, num_proc=8)
    print("pad_id, tokenizer         : {}".format(tokenizer.pad_token_id))
    print("pad_id, generation_config : {}".format(generation_config.pad_token_id))
    eval_dataset = None
    if data_args.eval_manifest_files is not None:
        eval_dataset = load_text_instruction_datasets(data_args, tokenizer=tokenizer, num_proc=8, do_eval=True)
    data_collator = TextInstructionDataCollator(pad_id=tokenizer.pad_token_id, padding_side=model_args.padding_side)
    print("For use_cache, must      padding_side=left ")
    logger.info("train dataset ready")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # 7. Initialize Trainer
    model.to('cuda')
    # print("starting memory tracking...")
    # torch.cuda.memory._record_memory_history(enabled=True, trace_alloc_record_context=True, _enable_expensive_cpp=True)
    # print("starting memory tracking...ok")
    # print("callbacks : {}".format())
    # training_args.include_inputs_for_metrics=True
    training_args.do_eval = True
    def compute_metrics(pred):
        import pdb; pdb.set_trace()
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            # 'Loss': loss,
            'Accuracy': acc,
        }

    trainer = llamaTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        # tokenizer=tokenizer,
    )
    logger.info("trainer ready")

    # 8. Training
    model.set_groups_used([0, 1])  ## only first group experts will used in training
    if training_args.do_train:
        # import pdb;pdb.set_trace()
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logger.info("resume training from ckpt")
            train_result = trainer.train(resume_from_checkpoint=True)
        else:
            logger.info("start training")
            train_result = trainer.train()
            
        if trainer.is_deepspeed_enabled:
            trainer.save_model()
        else:
            trainer_save_model_safe(trainer)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Save model
    if training_args.save_final_ckpt:
        logger.info("training finished, dumping model")
        model.config.use_cache = False  ## True
        trainer.save_state()  # for debug, not save
        if trainer.is_deepspeed_enabled:
            trainer.save_model()
        else:
            trainer_save_model_safe(trainer)
        tokenizer.save_pretrained(training_args.output_dir)
        generation_config.save_pretrained(training_args.output_dir)

    logger.info("🎉 All done~")


if __name__ == "__main__":
    train()
