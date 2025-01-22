import math
import os.path
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import init
from transformers.modeling_utils import dtype_byte_size

from smoe.models.mixtral_2group.configuration_mixtral2group import Mixtral2GroupConfig
from smoe.models.mixtral_2group.modeling_mixtral2group import Mixtral2GroupForCausalLM
from smoe.utils.io import dump_json, load_json


def is_safetensors_file(filepath):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    file_name = filepath.name
    if "1B" in str(filepath):
        print("model is 1B")
        return file_name == "model.safetensors"
    return re.match(r"model-\d{5}-of-\d{5}.safetensors", file_name) is not None


FFN_TYPE_MAP = {
    "modulelist": {
        "gate": "w1",
        "down": "w2",
        "up": "w3",
    },
}


def convert_safetensors(
    model_dir,
    dump_dir,
    # num_experts: int,
    num_experts_group0: int = 4,
    num_experts_group1: int = 4,
    top_k: int = 1,
    scale_factor: float = 1.0,
    num_moe_insert_layers: int = 4,  ## per 4 layers to insert MoE MLP
    moe_type: str = "modulelist",
    use_fft: bool = True,
    neuron_indices: dict = None,
    gate_weights: dict = None,
):
    # fmt: off
    model_folder = Path(model_dir)
    dump_folder = Path(dump_dir)
    dump_folder.mkdir(parents=True, exist_ok=True)
    print("dump_folder: {}".format(dump_folder))
    ffn_type_map = FFN_TYPE_MAP[moe_type]
    print("num_experts_group0: {}".format(num_experts_group0))
    print("num_experts_group1: {}".format(num_experts_group1))
    print("top_k: {}".format(top_k))
    print("num_moe_insert_layers: {}".format(num_moe_insert_layers))

    raw_total_size = -1   ##
    tensor_filepaths = []
    for filepath in model_folder.glob("*"):
        # print("filepath: \n {}".format(filepath))
        if not os.path.isdir(filepath):
            if is_safetensors_file(filepath):
                print("   the filepath is tensor weight load path  :) loading ....")
                tensor_filepaths.append(filepath)
            if filepath.name == "config.json":
                config = Mixtral2GroupConfig.from_pretrained(filepath)
                config.architectures = ["Mixtral2GroupForCausalLM"]
                config.num_experts_per_tok = top_k
                # config.num_local_experts = num_experts
                config.num_local_experts_group0 = num_experts_group0
                config.num_local_experts_group1 = num_experts_group1
                config.router_aux_loss_coef = 1e-2   ### 0.01 weight BL
                config.scale_factor = scale_factor
                config.moe_type = moe_type
                config.num_moe_insert_layers = num_moe_insert_layers
                config.use_fft = use_fft
                ##
                # config.intermediate_size = config.intermediate_size // num_experts
                print("intermediate_size: {}".format(config.intermediate_size))

                config.auto_map = {
                    "AutoConfig": "configuration_mixtral2group.Mixtral2GroupConfig",
                    "AutoModel": "modeling_mixtral2group.Mixtral2GroupModel",
                    "AutoModelForCausalLM": "modeling_mixtral2group.Mixtral2GroupForCausalLM",
                }
                ## save
                config.save_pretrained(dump_folder)
                for filename in [
                    "configuration_mixtral2group.py",
                    "modeling_mixtral2group.py",
                ]:
                    shutil.copy2(f"smoe/models/mixtral_2group/{filename}", dump_folder / filename)
                (dump_folder / "__init__.py").touch()
            elif filepath.name == "model.safetensors.index.json":
                raw_total_size = load_json(filepath)["metadata"]["total_size"]
            else:
                # cp to dump_dir
                shutil.copy2(filepath, dump_folder / filepath.name)

    router_records = set()
    weight_map = {}
    total_size = 0
    total_gate_size = 0  ##
    total_expert_size = 0
    visited_layers = set()
    for fi, filepath in enumerate(tensor_filepaths):
        with safe_open(filepath, framework="pt", device="cpu") as f:
            tensors = {}
            contained_layers = set()
            for key in f.keys():
                tensor = f.get_tensor(key)
                # print("key: {}\n {}".format(key, tensor))
                if ".mlp." in key:
                    print("mlp tensor key: {}\n {}".format(key, tensor.shape))
                    # preparation
                    layer_idx, ffn_type = re.search(
                        r"model.layers.(\d+).mlp.(gate|up|down)_proj.weight", key
                    ).groups()
                    layer_idx = int(layer_idx)

                    # is_moe = (layer_idx >= num_moe_contract_layers) and (layer_idx < config.num_hidden_layers - num_moe_contract_layers)
                    is_moe = ((layer_idx + 1) % num_moe_insert_layers == 0)

                    if is_moe:
                        contained_layers.add(layer_idx)
                        visited_layers.add(layer_idx)

                        if ffn_type == "down":
                            hsz, mid = tensor.shape
                            mid_idx = 1
                        else:
                            mid, hsz = tensor.shape
                            mid_idx = 0

                        # initialize gate weights
                        if layer_idx not in router_records:
                            if use_fft: ## 拼接
                                gate0_weight = torch.zeros(num_experts_group0, hsz * 2)
                                gate1_weight = torch.zeros(num_experts_group1, hsz * 2)
                            else:
                                gate0_weight = torch.zeros(num_experts_group0, hsz)
                                gate1_weight = torch.zeros(num_experts_group1, hsz)

                            init.kaiming_uniform_(gate0_weight, a=math.sqrt(5))
                            tensors[
                                f"model.layers.{layer_idx}.block_mlp_moe.groups.0.gate.weight"
                            ] = gate0_weight

                            init.kaiming_uniform_(gate1_weight, a=math.sqrt(5))
                            tensors[
                                f"model.layers.{layer_idx}.block_mlp_moe.groups.1.gate.weight"
                            ] = gate1_weight

                            router_records.add(layer_idx)
                        # new_ffn_type = ffn_type_map[ffn_type]

                        # initialize expert weights
                        for expert_idx in range(num_experts_group0):
                            tensors[
                                f"model.layers.{layer_idx}.block_mlp_moe.groups.0.experts.{expert_idx}.{ffn_type}_proj.weight"
                                ] = tensor.clone()
                        for expert_idx in range(num_experts_group1):
                            tensors[
                                f"model.layers.{layer_idx}.block_mlp_moe.groups.1.experts.{expert_idx}.{ffn_type}_proj.weight"
                                ] = tensor.clone()

                    else: ## 不是moe
                        tensors[key] = tensor

                else:
                    tensors[key] = tensor

            for key in tensors:
                tensors[key] = tensors[key].contiguous()
            save_file(tensors, dump_folder / filepath.name, metadata={"format": "pt"})
            for key, tensor in tensors.items():
                weight_size = tensor.numel() * dtype_byte_size(tensor.dtype)
                total_size += weight_size
                weight_map[key] = filepath.name
                if ".block_mlp_moe.groups" in key and ".gate." in key:
                    total_gate_size += weight_size
                if ".block_mlp_moe.groups" in key and "experts" in key:
                    total_expert_size +=weight_size
                print(key, tensor.shape)

    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    dump_json(index, dump_folder / "model.safetensors.index.json", indent=2)
    print("all router layers: {}".format(router_records))
    print("raw_total_size:    {}".format(raw_total_size))
    print("total_size:        {}".format(total_size))
    print("total_gate_size:   {}".format(total_gate_size))
    print("total_expert_size: {}".format(total_expert_size))
    print("total_expert_size / (num_experts_group0 + num_experts_group1): {}".format(total_expert_size / (num_experts_group0 + num_experts_group1)))
    print("total_size - total_gate_size - total_expert_size: {}".format(total_size - total_gate_size - total_expert_size))
    now_sizie = total_size - total_gate_size - total_expert_size + total_expert_size / (num_experts_group0 + num_experts_group1)
    print("total_size - total_gate_size - total_expert_size + total_expert_size / (num_experts_group0 + num_experts_group1): {}".format(now_sizie))
    print("gap_size:          {}".format(raw_total_size - now_sizie))
    # assert total_size - total_gate_size - total_expert_size + total_expert_size / (num_experts_group0 + num_experts_group1) == raw_total_size


if __name__ == "__main__":
    # num_experts = 8
    # # top_k = 2
    num_experts_group0 = 4
    num_experts_group1 = 4
    top_k = 1
    num_moe_insert_layers = 4

    # src_model_dir = "/home/zhanglinlin/models/llama/Meta-Llama-3.1-8B-Instruct"
    # tgt_model_dir_prefix = f"/home/zhanglinlin/pro/MoE/moe_mt/converted_models/Llama3.1-8B_base-2group-Top{top_k}_insert32"

    src_model_dir = "/home/zhanglinlin/models/llama/Meta-Llama-3.2-1B-Instruct"
    tgt_model_dir_prefix = f"/home/zhanglinlin/pro/MoE/moe_mt/converted_models/Llama3.2-1B_base-2group-Top{top_k}"

    tgt_moe_types = ["modulelist"]
    moe_type = "modulelist"

    neuron_indices_file = ""
    gate_weights_file = ""

    print(f"converting {moe_type}")
    convert_safetensors(
        src_model_dir,
        f"{tgt_model_dir_prefix}",
        num_experts_group0=num_experts_group0,
        num_experts_group1=num_experts_group1,
        top_k=top_k,
        num_moe_insert_layers=num_moe_insert_layers,
        moe_type=moe_type,
        neuron_indices=None
        if neuron_indices_file == ""
        else torch.load(neuron_indices_file),
        gate_weights=None if gate_weights_file == "" else torch.load(gate_weights_file),
    )

    print(f"testing {moe_type}")
    m = Mixtral2GroupForCausalLM.from_pretrained(f"{tgt_model_dir_prefix}").bfloat16()

    print(f"Re-saving {moe_type}")
    m.save_pretrained(f"{tgt_model_dir_prefix}")

    print("Done")
    # fmt: on
