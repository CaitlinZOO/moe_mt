import argparse

import torch

from smoe.utils.expert_construction.convert_llama_to_mixtral_2group import (
    convert_safetensors,
)

# fmt: off
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--neuron_indices_file', type=str, default=None)
    parser.add_argument('--gate_weights_file', type=str, default=None)

    parser.add_argument('--num_experts_group0', type=int, default=4)
    parser.add_argument('--num_experts_group1', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--scale_factor', type=float, default=1.0)
    parser.add_argument('--num_moe_insert_layers', type=int, default=4)
    parser.add_argument('--moe_implementation_type', type=str, default='modulelist', choices=["modulelist"])
    parser.add_argument('--use_fft', type=bool, default=True)

    args = parser.parse_args()
    print(args, "\n")

    convert_safetensors(
        args.model_path,
        args.save_path,
        num_experts_group0=args.num_experts_group0,
        num_experts_group1=args.num_experts_group1,
        top_k=args.top_k,
        scale_factor=args.scale_factor,
        num_moe_insert_layers=args.num_moe_insert_layers,
        moe_type=args.moe_implementation_type,
        use_fft=args.use_fft,
        neuron_indices=None if args.neuron_indices_file is None else torch.load(args.neuron_indices_file),
        gate_weights=None if args.gate_weights_file is None else torch.load(args.gate_weights_file),
    )
    print("Done!")
# fmt: on
