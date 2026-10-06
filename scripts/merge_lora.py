"""把 LoRA 合并进基座模型，得到一个完整的模型目录（之后转 GGUF / 导入 Ollama 用）。

  python scripts/merge_lora.py models/Qwen3-1.7B adapters/v2/raft_1.7b models/raft-1.7b-merged

合并 = 把 LoRA 的两个小矩阵乘起来（B·A），加回原权重 W。合并后推理不再需要 peft，速度和原模型一样。
"""
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, adapter, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)   # 用训练时保存的 tokenizer
print("merged →", out)
