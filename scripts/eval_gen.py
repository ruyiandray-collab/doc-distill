"""【在 AutoDL 上运行】用 vLLM 让各个模型回答 raft_test，结果写到 preds/<name>.jsonl

同一个基座只加载一次，原版和各个 LoRA 用 LoRARequest 切换（vLLM 的多 LoRA 服务能力）。
除了答案，还记录两种速度：
  - 批量吞吐：75 题一起跑的总耗时 / 输出 tokens 每秒
  - 单条延迟：逐条跑前 10 题的平均耗时（更接近一个用户在问答页面上的体验）
"""
import json, os, sys, time
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

W = Path("/root/autodl-tmp/work")
TEST = json.loads((W / "data/raft_test.json").read_text(encoding="utf-8"))
OUT = Path(os.getenv("PRED_DIR", str(W / "preds"))); OUT.mkdir(exist_ok=True)
CKPT = os.getenv("CKPT_DIR", "/root/autodl-tmp/ckpt")       # v2 用 /root/autodl-tmp/ckpt/v2
SKIP_BASE = os.getenv("SKIP_BASE") == "1"                     # 原版模型不受模板影响，v2 不用重测
SP = SamplingParams(temperature=0, max_tokens=512)
GROUPS = {
    "8b":   ("/root/autodl-tmp/models/Qwen3-8B",   {"base_8b": None, "raft_8b": "raft_8b", "plain_8b": "plain_8b"}),
    "1.7b": ("/root/autodl-tmp/models/Qwen3-1.7B", {"base_1.7b": None, "raft_1.7b": "raft_1.7b"}),
}

def msgs(x):
    return [{"role": "system", "content": x["system"]}, {"role": "user", "content": x["instruction"]}]

def run(llm, name, lora):
    kw = dict(sampling_params=SP, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    if lora:
        kw["lora_request"] = LoRARequest(lora, hash(lora) % 10000 + 1, f"{CKPT}/{lora}")
    t0 = time.time(); outs = llm.chat([msgs(x) for x in TEST], **kw); batch_s = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    lat = []
    for x in TEST[:10]:
        t = time.time(); llm.chat([msgs(x)], **kw); lat.append(time.time() - t)
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for x, o in zip(TEST, outs):
            f.write(json.dumps({"meta": x["meta"], "ref": x["output"], "pred": o.outputs[0].text}, ensure_ascii=False) + "\n")
    stat = {"name": name, "batch_s": round(batch_s, 1), "out_tok_per_s": round(ntok / batch_s), "avg_latency_s": round(sum(lat) / len(lat), 2)}
    print(json.dumps(stat, ensure_ascii=False))
    with (OUT / "speed.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(stat, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    for g in (sys.argv[1:] or GROUPS):
        path, runs = GROUPS[g]
        llm = LLM(path, enable_lora=True, max_lora_rank=16, max_model_len=6144, gpu_memory_utilization=0.85)
        for name, lora in runs.items():
            if lora or not SKIP_BASE:
                run(llm, name, lora)
        del llm
