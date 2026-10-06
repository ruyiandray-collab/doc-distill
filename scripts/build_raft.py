"""把 qa_raw.jsonl 组装成 RAFT 训练/测试集（LLaMA-Factory 的 alpaca 格式）

每条样本 = 问题 + 4 段资料（顺序打乱）：
  - 80%：1 段正确原文（oracle）+ 3 段干扰（2 段 BM25 最像的"难干扰" + 1 段随机）
  - 20%：4 段全是干扰 → 标准答案是拒答，教模型"资料里没有就说没有"
目标输出 = 依据（标出第几段资料 + 逐字引用）+ 回答。

划分：按"块"划分，测试题对应的块不会作为 oracle 出现在训练集里，防止模型靠背答案拿分。
同时输出一份不带干扰的对照训练集（plain），用于消融实验：干扰训练到底有没有用。
"""
import asyncio, json, os, random, re
from collections import defaultdict
from pathlib import Path
import jieba
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
SEED, TEST_RATIO, P_ORACLE, N_DOCS = 42, 0.15, 0.8, 4

SYSTEM = ("你是仓储与安全生产法规问答助手。只能依据用户提供的资料回答。"
          "先写「依据」：注明是第几段资料并逐字引用关键原文；再写「回答」，结论在前。"
          "如果资料中没有能回答问题的内容，直接说明无法回答，不要编造。")
REFUSAL = "依据：提供的资料中没有与该问题直接相关的规定。\n回答：根据现有资料无法回答这个问题，建议查阅相关法规原文或咨询安全管理部门。"

norm = lambda s: re.sub(r"\s+", "", s)
tok = lambda s: [w for w in jieba.lcut(s) if w.strip()]

def fmt_input(question, docs):
    parts = [f"[资料{i + 1}]《{c['doc']}》\n{c['text']}" for i, c in enumerate(docs)]
    return "资料：\n" + "\n\n".join(parts) + f"\n\n问题：{question}"

CHECK = """下面是若干段资料和一个问题。请严格判断：仅凭这些资料，能否给出该问题的明确答案（不允许用资料以外的常识补充）？
{inp}

只输出 JSON：{{"answerable": true 或 false}}"""

async def answerable(inputs):
    client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])
    sem = asyncio.Semaphore(10)
    async def one(inp):
        async with sem:
            r = await client.chat.completions.create(model="deepseek-chat", temperature=0, response_format={"type": "json_object"},
                                                     messages=[{"role": "user", "content": CHECK.format(inp=inp)}])
            return bool(json.loads(r.choices[0].message.content).get("answerable"))
    return await asyncio.gather(*[one(i) for i in inputs])

def main():
    random.seed(SEED)
    jieba.setLogLevel(60)
    chunks = [json.loads(l) for l in (D / "chunks.jsonl").open(encoding="utf-8")]
    byid = {c["id"]: c for c in chunks}
    qa = [q for q in map(json.loads, (D / "qa_raw.jsonl").open(encoding="utf-8")) if q["quote_ok"]]

    # 按块划分：每份文档各抽 15% 的块做测试
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["doc"]].append(c["id"])
    test_chunks = set()
    for ids in by_doc.values():
        test_chunks |= set(random.sample(ids, max(1, round(len(ids) * TEST_RATIO))))

    bm25 = BM25Okapi([tok(c["text"]) for c in chunks])

    def distractors(q, k_hard=2, k_rand=1, exclude=(), other_doc=False):
        scores = bm25.get_scores(tok(q["question"]))
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        # 含答案的不能当干扰；拒答样本只从"别的文档"取干扰，否则同一文档的相邻条款常常也能回答，标签就错了
        bad = lambda c: (c["id"] in exclude or norm(q["quote"]) in norm(c["text"])
                         or (other_doc and c["doc"] == q["doc"]))
        hard = [chunks[i] for i in order if not bad(chunks[i])][:k_hard]
        pool = [c for c in chunks if not bad(c) and c not in hard]
        return hard + random.sample(pool, k_rand)

    raft = {"train": [], "test": []}
    plain = []
    for q in qa:
        split = "test" if q["chunk_id"] in test_chunks else "train"
        oracle = byid[q["chunk_id"]]
        # 训练集的干扰段不能用测试块，避免测试块内容泄漏进训练
        excl = {oracle["id"]} | (test_chunks if split == "train" else set())
        with_oracle = random.random() < P_ORACLE
        docs = distractors(q, exclude=excl, other_doc=not with_oracle) + ([oracle] if with_oracle else [])
        if not with_oracle:
            docs += distractors(q, k_hard=0, k_rand=1, exclude=excl | {d["id"] for d in docs}, other_doc=True)
        random.shuffle(docs)
        if with_oracle:
            k = docs.index(oracle) + 1
            out = f"依据：[资料{k}]《{oracle['doc']}》：「{q['quote']}」\n回答：{q['answer']}"
        else:
            out = REFUSAL
        raft[split].append({"system": SYSTEM, "instruction": fmt_input(q["question"], docs), "input": "", "output": out,
                            "meta": {"chunk_id": q["chunk_id"], "type": q["type"], "has_oracle": with_oracle}})
        if split == "train":   # 对照组：只给正确原文，没有干扰、没有拒答
            plain.append({"system": SYSTEM, "instruction": fmt_input(q["question"], [oracle]), "input": "",
                          "output": f"依据：[资料1]《{oracle['doc']}》：「{q['quote']}」\n回答：{q['answer']}"})

    # 二次校验：让 DeepSeek 判断拒答样本的 4 段资料是否其实能回答；能回答的标签有误，丢掉
    for s in ("train", "test"):
        neg = [r for r in raft[s] if not r["meta"]["has_oracle"]]
        flags = asyncio.run(answerable([r["instruction"] for r in neg]))
        bad = {id(r) for r, f in zip(neg, flags) if f}
        raft[s] = [r for r in raft[s] if id(r) not in bad]
        print(f"{s}: 拒答样本 {len(neg)} 条，其中 {len(bad)} 条资料其实能回答，已剔除")

    strip = lambda rows:[{k: v for k, v in r.items() if k != "meta"} for r in rows]   # 训练文件不带 meta
    for name, rows in [("raft_train", strip(raft["train"])), ("raft_test", raft["test"]), ("plain_train", plain)]:
        (D / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    info = {n: {"file_name": f"{n}.json", "columns": {"system": "system", "prompt": "instruction", "query": "input", "response": "output"}}
            for n in ("raft_train", "raft_test", "plain_train")}
    (D / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")

    for s in ("train", "test"):
        r = raft[s]; ref = sum(not x["meta"]["has_oracle"] for x in r)
        print(f"RAFT {s}: {len(r)} 条（其中拒答 {ref} 条，{ref / len(r):.0%}），涉及 {len({x['meta']['chunk_id'] for x in r})} 块")
    print(f"plain train: {len(plain)} 条；最长样本 {max(len(x['instruction']) for x in raft['train'])} 字")

if __name__ == "__main__":
    main()
