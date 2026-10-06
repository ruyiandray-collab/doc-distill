"""【在 AutoDL 上运行】检查"训练时的提示词"和"推理时的提示词"在 token 层面是否完全一致。

训练侧：LLaMA-Factory 的模板把一条样本编码成 prompt_ids；
推理侧：vLLM 用模型自带的 chat template（enable_thinking=False）。
两者的 prompt 必须逐 token 相同，否则模型推理时看到的是训练时没见过的开头。
"""
import json
from transformers import AutoTokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

MODEL = "/root/autodl-tmp/models/Qwen3-1.7B"
x = json.load(open("/root/autodl-tmp/work/data/raft_train.json", encoding="utf-8"))[0]
msgs = [{"role": "user", "content": x["instruction"]}, {"role": "assistant", "content": x["output"]}]

tok = AutoTokenizer.from_pretrained(MODEL)
infer = tok.apply_chat_template([{"role": "system", "content": x["system"]}, msgs[0]],
                                add_generation_prompt=True, enable_thinking=False, tokenize=False)
infer_ids = tok(infer, add_special_tokens=False)["input_ids"]

for name, kw in [("v1 qwen3_nothink", dict(template="qwen3_nothink")),
                 ("v2 qwen3 + enable_thinking=false", dict(template="qwen3", enable_thinking=False))]:
    template = get_template_and_fix_tokenizer(tok, DataArguments(**kw))
    prompt_ids, resp_ids = template.encode_oneturn(tok, msgs, x["system"])
    same = prompt_ids == infer_ids
    print(f"{name}: prompt 与推理一致 = {same}")
    if not same:
        print("   训练 prompt 结尾:", repr(tok.decode(prompt_ids[-8:])))
        print("   推理 prompt 结尾:", repr(tok.decode(infer_ids[-8:])))
    print("   训练目标开头:", repr(tok.decode(resp_ids[:6])))
