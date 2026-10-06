"""本地 Ollama（CPU）跑 raft_test → preds/<name>.jsonl，用来看 Q4 量化掉了多少分、笔记本 CPU 上有多慢。

  python scripts/eval_ollama.py raft-1.7b raft_1.7b_v2_q4_ollama
"""
import json, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
model, name = sys.argv[1], sys.argv[2]
TEST = json.loads((ROOT / "data/raft_test.json").read_text(encoding="utf-8"))

def chat(x):
    # think=False：Ollama 用 GGUF 自带的 Qwen3 模板，不关思考的话提示词里没有空 <think> 块（与训练不一致），
    # 而且模型的输出会被当成"思考"放进 thinking 字段，content 为空——第一次评测 16/71 题就是这样被判错的
    body = {"model": model, "stream": False, "think": False, "keep_alive": "30m", "options": {"temperature": 0, "num_predict": 512},
            "messages": [{"role": "system", "content": x["system"]}, {"role": "user", "content": x["instruction"]}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

PRED = ROOT / "preds" / f"{name}.jsonl"
done = sum(1 for _ in PRED.open(encoding="utf-8")) if PRED.exists() else 0   # 断点续跑：已答的题跳过
stats, out = [], PRED.open("a", encoding="utf-8")
for i, x in enumerate(TEST):
    if i < done:
        continue
    t = time.time(); r = chat(x); dt = time.time() - t
    out.write(json.dumps({"meta": x["meta"], "ref": x["output"], "pred": r["message"]["content"]}, ensure_ascii=False) + "\n"); out.flush()
    stats.append((dt, r["prompt_eval_count"], r.get("prompt_eval_duration", 1) / 1e9, r["eval_count"], r["eval_duration"] / 1e9))
    print(f"{i + 1}/{len(TEST)} {dt:.1f}s", flush=True)
n = len(stats)
if not n:
    sys.exit("没有新题可跑")
summary = {"name": name, "n_timed": n, "avg_latency_s": round(sum(s[0] for s in stats) / n, 1),
           "prefill_tok_per_s": round(sum(s[1] for s in stats) / sum(s[2] for s in stats)),
           "decode_tok_per_s": round(sum(s[3] for s in stats) / sum(s[4] for s in stats), 1)}
(ROOT / "results" / f"speed_{name}.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
print(summary)
