"""对 tree-RAG 的检索结果做重排序，比较"正确原文进入前 4 段"的比例（只看检索，不调用回答模型，很便宜）

  python scripts/rerank.py              # 读 results/e2e/retrieval_tree.jsonl
输出：results/e2e/retrieval_tree+bm25.jsonl、retrieval_tree+bge.jsonl（命中段已按新顺序排好）
      results/e2e/rerank_summary.json

为什么要重排：tree-RAG 每题平均返回 12.6 段，顺序是"文档 → 章节 → 页"的遍历顺序，
不是按和问题的相关度排的；而蒸馏模型只吃前 4 段（训练时就是 4 段）。
bge-reranker 是交叉编码器：把"问题 + 段落"拼在一起送进一个 BERT 类模型直接打相关分，
比"问题向量 vs 段落向量"的双塔检索更准，但每对都要算一遍，所以只适合给少量候选重排。
"""
import json, re, time
from pathlib import Path
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
E2E = ROOT / "results" / "e2e"
jieba.setLogLevel(60)
norm = lambda s: re.sub(r"[\s，。、；：,.;:“”\"'「」『』()（）]", "", s)
tok = lambda s: [w for w in jieba.lcut(s) if w.strip()]

test = json.loads((ROOT / "data/raft_test.json").read_text(encoding="utf-8"))
qa = {q["question"]: q for q in map(json.loads, (ROOT / "data/qa_raw.jsonl").open(encoding="utf-8"))}
ret = [json.loads(l) for l in (E2E / "retrieval_tree.jsonl").open(encoding="utf-8")]

def gold_rank(r, hits):
    """正确原文出现在第几段（1 开始）；没检索到返回 None"""
    q = norm(qa[r["question"]]["quote"])[:40]
    return next((i + 1 for i, h in enumerate(hits) if q in norm(h["text"])), None)

def bm25_order(r):
    hits = r["hits"]
    scores = BM25Okapi([tok(h["text"]) for h in hits]).get_scores(tok(r["question"]))
    return [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]

model = CrossEncoder(str(ROOT / "models/bge-reranker-v2-m3"), max_length=512)
def bge_order(r):
    scores = model.predict([(r["question"], h["text"]) for h in r["hits"]], batch_size=16)
    return [h for _, h in sorted(zip(scores, r["hits"]), key=lambda x: -x[0])]

summary = {}
for name, fn in [("tree", lambda r: r["hits"]), ("tree+bm25", bm25_order), ("tree+bge", bge_order)]:
    t0, out, ranks = time.time(), [], []
    for r in ret:
        hits = fn(r)
        ranks.append(gold_rank(r, hits))
        out.append({**r, "hits": hits})
    secs = (time.time() - t0) / len(ret)
    if name != "tree":
        with (E2E / f"retrieval_{name}.jsonl").open("w", encoding="utf-8") as f:
            for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    n = len(ranks)
    summary[name] = {f"gold_in_top{k}": round(sum(1 for x in ranks if x and x <= k) / n, 3) for k in (1, 2, 4, 8)}
    summary[name]["gold_in_all"] = round(sum(1 for x in ranks if x) / n, 3)
    summary[name]["rerank_s_per_q"] = round(secs, 2)
    print(name, summary[name], flush=True)
(E2E / "rerank_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
