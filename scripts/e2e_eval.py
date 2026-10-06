"""端到端评测：tree-RAG 真实检索 + 不同回答模型（在 mczc-tree-rag 的虚拟环境里运行）

与 raft_test 的区别：那里的 4 段资料是"人为配好"的（必有/必无正确原文）；
这里由 tree-RAG 自己去 7 份法规里检索，检索错了回答模型也救不回来，更接近真实使用。

同一题只检索一次，结果缓存后给所有回答模型共用——保证比较的只是"回答模型"这一个变量。
回答系统：
  deepseek_default  原 tree-RAG：DeepSeek + 原提示词 + 全部命中段
  deepseek_raft     DeepSeek + RAFT 格式（前 4 段，每段截 1000 字）——隔离"格式/截断"的影响
  local_raft_1.7b   本地 Ollama 的蒸馏 1.7B（Q4）+ RAFT 格式——完全离线的回答
指标：检索命中（标准答案的原文是否进入前 4 段 / 全部命中段）、回答正确率（DeepSeek 裁判）、平均耗时

用法（环境变量见 README）：python scripts/e2e_eval.py [--systems a,b] [--limit N]
"""
import argparse, asyncio, json, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("TREERAG_DIR", str(ROOT.parent / "mczc-tree-rag")))
os.environ["RERANK"] = "none"   # 重排由 rerank.py 预先做好（--mode tree+bm25 / tree+bge），这里不能再排一次
from openai import AsyncOpenAI                       # noqa: E402
from treerag import config                           # noqa: E402
from treerag.answer import SYSTEM_PROMPT, format_context, _raft_messages   # noqa: E402
from treerag.retriever import retrieve               # noqa: E402

OUT = ROOT / "results" / "e2e"; OUT.mkdir(parents=True, exist_ok=True)
norm = lambda s: re.sub(r"[\s，。、；：,.;:“”\"'「」『』()（）]", "", s)
deepseek = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=config.LLM_API_KEY)
ollama = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=900)

SYSTEMS = {
    "deepseek_default": (deepseek, "deepseek-chat", "default"),
    "deepseek_raft":    (deepseek, "deepseek-chat", "raft"),
    "local_raft_1.7b":  (ollama,   "raft-1.7b",     "raft"),
}

def load_questions():
    test = json.loads((ROOT / "data/raft_test.json").read_text(encoding="utf-8"))
    qa = {q["question"]: q for q in map(json.loads, (ROOT / "data/qa_raw.jsonl").open(encoding="utf-8"))}
    rows = []
    for x in test:
        q = x["instruction"].split("问题：")[-1]
        rows.append({"question": q, "answer": qa[q]["answer"], "quote": qa[q]["quote"], "doc": qa[q]["doc"]})
    return rows

async def get_retrieval(rows, mode):
    f = OUT / f"retrieval_{mode}.jsonl"
    if "+" in mode and not f.exists():
        sys.exit(f"{f.name} 不存在：先运行 scripts/rerank.py")
    cache ={r["question"]: r for r in map(json.loads, f.open(encoding="utf-8"))} if f.exists() else {}
    sem = asyncio.Semaphore(4)
    async def one(r):
        if r["question"] in cache: return
        async with sem:
            t = time.time(); res = await retrieve(r["question"], mode)
            hits = [{"name": h["name"], "location": h.get("location", ""), "title": h.get("title", ""), "text": h["text"]} for h in res["hits"]]
            cache[r["question"]] = {"question": r["question"], "hits": hits, "retrieve_s": round(time.time() - t, 1)}
    await asyncio.gather(*[one(r) for r in rows])
    # 写回全部缓存（不只是本次的 rows）：--limit 跑子集时不能把已有的完整缓存冲掉
    tmp = f.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fo:
        for v in cache.values(): fo.write(json.dumps(v, ensure_ascii=False) + "\n")
    tmp.replace(f)
    return cache

def messages(fmt, q, hits):
    if fmt == "raft":
        return _raft_messages(q, hits)
    return [{"role": "system", "content": SYSTEM_PROMPT.format(context=format_context(hits))}, {"role": "user", "content": q}]

async def run_system(name, rows, ret, mode):
    client, model, fmt = SYSTEMS[name]
    # mode=tree 时沿用老文件名；tree+bm25 / tree+bge 是 rerank.py 生成的重排结果
    f = OUT / (f"answers_{name}.jsonl" if mode == "tree" else f"answers_{mode}_{name}.jsonl")
    done = {r["question"]: r for r in map(json.loads, f.open(encoding="utf-8"))} if f.exists() else {}
    sem = asyncio.Semaphore(1 if client is ollama else 8)      # 本地 CPU 一次只跑一题
    async def one(r):
        if r["question"] in done: return
        async with sem:
            t = time.time()
            extra = {"extra_body": {"reasoning_effort": "none"}} if client is ollama else {}   # 关闭 Qwen3 思考，见 eval_ollama.py
            resp = await client.chat.completions.create(model=model, temperature=0, max_tokens=600,
                                                        messages=messages(fmt, r["question"], ret[r["question"]]["hits"]), **extra)
            done[r["question"]] = {"question": r["question"], "pred": resp.choices[0].message.content, "answer_s": round(time.time() - t, 1)}
            with f.open("a", encoding="utf-8") as fo: fo.write(json.dumps(done[r["question"]], ensure_ascii=False) + "\n")
            print(f"  {name} {len(done)}/{len(rows)}", flush=True)
    await asyncio.gather(*[one(r) for r in rows])
    return done

JUDGE = """你是严格的阅卷老师。问题：{q}
参考答案：{ref}
学生回答：{pred}
结论与关键数值/条件一致为 correct；结论对但遗漏重要条件或有小错为 partial；结论错误、答非所问、或在有答案时声称资料中找不到为 wrong。
只输出 JSON：{{"verdict": "correct|partial|wrong"}}"""

async def judge(rows, ans):
    sem = asyncio.Semaphore(10)
    async def one(r):
        async with sem:
            resp = await deepseek.chat.completions.create(model="deepseek-chat", temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": JUDGE.format(q=r["question"], ref=r["answer"], pred=ans[r["question"]]["pred"])}])
            return {"correct": 1, "partial": 0.5}.get(json.loads(resp.choices[0].message.content).get("verdict"), 0)
    return await asyncio.gather(*[one(r) for r in rows])

async def main(a):
    rows = load_questions()[: a.limit or None]
    ret = await get_retrieval(rows, a.mode)
    gold = lambda r, k: norm(r["quote"])[:40] in norm("".join(h["text"] for h in ret[r["question"]]["hits"][:k]))
    summary = {"n": len(rows), "mode": a.mode,
               "gold_in_top4": sum(gold(r, 4) for r in rows) / len(rows),
               "gold_in_all_hits": sum(gold(r, 99) for r in rows) / len(rows),
               "avg_hits": sum(len(ret[r["question"]]["hits"]) for r in rows) / len(rows),
               "avg_retrieve_s": sum(ret[r["question"]]["retrieve_s"] for r in rows) / len(rows)}
    if a.limit:   # 子集只用来调试，不覆盖完整评测的 summary
        a.mode_tag = f"{a.mode}_limit{a.limit}"
    for name in a.systems.split(","):
        ans = await run_system(name, rows, ret, a.mode)
        s = await judge(rows, ans)
        summary[name] = {"acc": sum(s) / len(s), "avg_answer_s": sum(ans[r["question"]]["answer_s"] for r in rows) / len(rows),
                         "acc_when_gold_in_top4": sum(x for x, r in zip(s, rows) if gold(r, 4)) / max(1, sum(gold(r, 4) for r in rows))}
    (OUT / f"summary_{getattr(a, 'mode_tag', a.mode)}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tree")
    ap.add_argument("--systems", default="deepseek_default,deepseek_raft,local_raft_1.7b")
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
