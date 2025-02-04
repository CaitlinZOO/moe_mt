<div align="center">
  <h1>LLaMA-MoE v2: Exploring Sparsity of LLaMA from Perspective of Mixture-of-Experts with Post-Training</h1>
  <img src="docs/imgs/title-favicon.png" width="200" alt="LLaMA-MoE favicon" style="border-radius: 5%;"><br />
  <span style="color:red">📢 <strong><i>A SMALLER AFFORDABLE MoE MODEL FOR EVERYONE!!</i></strong></span>
  <div>
    <a href="https://huggingface.co/LLaMA-MoE-v2" target="_blank">🤗 Model Weights</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#installation">⚙️ Installation Guide</a> | <a href="#expert-construction">🚧 Expert Construction</a> | <a href="#sft">💬 Supervised Fine-Tuning (SFT)</a> | <a href="#evaluation">💎 Evaluation</a>  <br /> 
    <a href="https://arxiv.org/pdf/2411.15708" target="_blank" style="display: inline-block; margin-top: 10px;"> 📃 Technical Report </a>
  </div>
</div>



<h2 id="quick-start">🚀 QuickStart</h2>

```python
# python>=3.10

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "LLaMA-MoE-v2/LLaMA-MoE-v2-3_5B-2_8"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.to("cuda:0")

input_text = "Suzhou is famous for?"

input_text = f"<|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda:0")

pred = model.generate(**inputs, max_length=50, temperature=0.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```



<h2 id="installation">⚙️ Installation</h2>

1. Prepare conda environment: `conda create -n smoe python=3.11` (If your environment name is not `smoe`, you may need to change environment in launching scripts)
2. Add correct environment variables in `~/.bashrc` (`gcc` is set to newer version for installing `flash-attn`). e.g.:
     (如果服务器上有cuda驱动，不用这一步安装和写PATH)
    ```bash
    export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH
    export PATH=/mnt/petrelfs/share/gcc-10.1.0/bin:$PATH
    export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc-10.1.0/lib64:$LD_LIBRARY_PATH
    ```
3. Take the variables into effect: `source ~/.bashrc`
4. Install PyTorch (CUDA-11.8): `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
5. Install dependencies （requirements里的pkgs，确保transformers和flash-attn的版本，其他版本可以不一致）: `pip install -r requirements.txt`
6. Install `flash-attn`: `pip install flash-attn==2.6.1 --no-build-isolation`. You may need to follow the [flash-attn installation instructions](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) to avoid some errors.
7. Install the latest Git: `conda install git`
8. Clone the repo: `git@github.com:LLaMA-MoE/LLaMA-MoE-v2.git` (If you don't setup the ssh key to GitHub, you may not able to clone through ssh. Check the [docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) about it.)
9. Change current directory: `cd LLaMA-MoE-v2`
10. Install `smoe` in [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e): `pip install -e .[dev]`
11. Setup `pre-commit` hooks （可以不用设置）: `pre-commit install`




（先转换复制model）
- LLaMA-MoE 2group 4experts: `bash scripts/expert_construction/convert/convert_mixtral_2group_base.sh`

For more information, please refer to [Expert Construction docs](docs/expert_construction/README.md).


<h2 id="sft">💬 Supervised Fine-Tuning (SFT)</h2>

- 训练参数

   ```bash
   python train.py \
   --dataset_save_dir=${data_dir}    数据处理后保存的路径，如果已经有处理好的，直接加载
   --manifest_files=${dataset_dir_or_path}     数据全路径文件，多个文件用 | 隔开，目前只能是json或者csv文件
   --input_fields="src_text|src_text"     数据中指定的字段，作为gpt的输入
   --output_fields="src_text|tgt_text"     数据中指定的字段，作为gpt的生成
   --instructions="|"     指令，一个数据文件对应一个，用 | 隔开， lm任务是空，没有指令，翻译任务比如是  "Please translate the English text into Spanish: | Please     translate the English text into French: "
     
   ```

- sft stage_1  lm : `bash scripts/sft/sft_lm_2group_4e_top1_base.sh`
- sft stage_2  st : `bash scripts/sft/sft_mt_2group_4e_top1_base.sh`

- **NOTICE:** Please create `logs/` folder manually: `mkdir -p logs`

  We provide simple examples of SFT to build chatbots. Please refer to [SFT docs](docs/supervised_fine_tuning/LLaMA-MoE-v2.md) for more details.




<h2 id="citation">📑 Citation</h2>

```bibtex
@misc{llama-moe-v2,
  title={LLaMA-MoE v2: Exploring Sparsity of LLaMA from Perspective of Mixture-of-Experts with Post-Training},
  author={Xiaoye Qu, Daize Dong, Xuyang Hu, Tong Zhu, Weigao Sun, Yu Cheng},
  year={2024},
  month={Nov},
  url={https://arxiv.org/abs/2411.15708}
}
```

<hr>
<p align="center">LLaMA-MoE Team w/ ❤️</p>
