"""给 preds/*.jsonl 打分 → results/scores.json + 控制台表格

指标（75 题：有正确原文的 54 题 + 没有的 21 题，以实际数量为准）：
  答案正确率  有原文的题，DeepSeek 当裁判对照参考答案判 正确=1 / 部分正确=0.5 / 错误=0
  引用正确率  有原文的题，指出的资料编号对，且引号里的文字确实出自那段原文（≥8 字）
  误拒率      有原文的题，却回答"无法回答"
  拒答率      没有原文的题，正确地拒绝回答（越高越好，说明不编造）
  格式遵循率  输出里同时有「依据」和「回答」
裁判结果缓存在 results/judge_cache.json，重跑不重复花钱。
"""
import asyncio, hashlib, json, os, re
from pathlib import Path
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
PREDS, RES = ROOT / "preds", ROOT / "results"
CHUNKS = {c["id"]: c for c in map(json.loads, (ROOT / "data/chunks.jsonl").open(encoding="utf-8"))}
TEST = json.loads((ROOT / "data/raft_test.json").read_text(encoding="utf-8"))
QUESTION = [x["instruction"].split("问题：")[-1] for x in TEST]
CACHE_F = RES / "judge_cache.json"

norm = lambda s: re.sub(r"[\s，。、；：,.;:“”\"'「」『』()（）]", "", s)
REFUSE = re.compile(r"(无法|不能).{0,6}(回答|判断|作答|确定)|并未(涉及|规定|列出|提及)|没有(与该问题|相关|直接相关|提及|涉及)|未(提及|涉及|包含|找到)|不包含")

def is_refusal(t):
    """只看「回答」部分的开头：依据部分里出现"没有提及"之类的话不算拒答。"""
    m = list(re.finditer(r"回答[*\s]*[：:]", t))
    ans = t[m[-1].end():] if m else t
    return bool(REFUSE.search(ans[:120]))

def cite_ok(pred, ref, chunk_id):
    k = re.search(r"资料(\d)", ref).group(1)
    basis = pred.split("回答")[0]
    m = re.search(r"资料\s*(\d)", basis)
    if not m or m.group(1) != k:
        return False
    quotes = re.findall(r"[「“\"『]([^」”\"』]{8,})[」”\"』]", basis)
    src = norm(CHUNKS[chunk_id]["text"])
    return any(norm(q)[:60] in src for q in quotes)

JUDGE = """你是严格的阅卷老师。
问题：{q}
参考答案：{ref}
学生回答：{pred}
1. refused：学生是否在拒答，即最终没有给出实质结论，而是表示资料不足、无法回答？（只在依据里说"某段资料没提到"、但最后给出了结论的，不算拒答）
2. verdict：对照参考答案，结论与关键数值/条件一致为 correct；结论对但遗漏重要条件或有小错为 partial；结论错误、答非所问或拒答为 wrong。
只输出 JSON：{{"refused": true 或 false, "verdict": "correct|partial|wrong"}}"""
NO_ANSWER_REF = "（本题提供的资料中没有答案，正确做法是拒答）"

async def judge_all(items, cache):
    client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])
    sem = asyncio.Semaphore(10)
    async def one(key, q, ref, pred):
        if key in cache: return
        async with sem:
            r = await client.chat.completions.create(model="deepseek-chat", temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": JUDGE.format(q=q, ref=ref, pred=pred)}])
            cache[key] = json.loads(r.choices[0].message.content)
    await asyncio.gather(*[one(*it) for it in items])

def main():
    RES.mkdir(exist_ok=True)
    cache = json.loads(CACHE_F.read_text(encoding="utf-8")) if CACHE_F.exists() else {}
    runs = {p.stem: [json.loads(l) for l in p.open(encoding="utf-8")] for p in sorted(PREDS.glob("*.jsonl")) if p.stem != "speed"}
    todo = []
    for name, rows in runs.items():
        for i, r in enumerate(rows):
            ref_ans = r["ref"].split("回答：")[-1] if r["meta"]["has_oracle"] else NO_ANSWER_REF
            key = hashlib.md5((QUESTION[i] + r["pred"] + "v2").encode()).hexdigest()
            r["_key"] = key
            todo.append((key, QUESTION[i], ref_ans, r["pred"]))
    asyncio.run(judge_all(todo, cache))
    CACHE_F.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    score = {"correct": 1, "partial": 0.5, "wrong": 0}
    refused = lambda r: bool(cache[r["_key"]].get("refused"))
    table = {}
    for name, rows in runs.items():
        pos = [r for r in rows if r["meta"]["has_oracle"]]; neg = [r for r in rows if not r["meta"]["has_oracle"]]
        table[name] = {
            "答案正确率": sum(score.get(cache[r["_key"]].get("verdict"), 0) for r in pos) / len(pos),
            "引用正确率": sum(cite_ok(r["pred"], r["ref"], r["meta"]["chunk_id"]) for r in pos) / len(pos),
            "误拒率": sum(refused(r) for r in pos) / len(pos),
            "拒答率": sum(refused(r) for r in neg) / len(neg),
            "拒答率_正则": sum(is_refusal(r["pred"]) for r in neg) / len(neg),
            # 严格：开头有「依据」二字；结构：有资料编号引用（或拒答）且有「回答」——RAFT 模型常漏写「依据：」这两个字
            "格式遵循率": sum("依据" in r["pred"] and "回答" in r["pred"] for r in rows) / len(rows),
            "结构遵循率": sum(("资料" in r["pred"].split("回答")[0] or refused(r)) and "回答" in r["pred"] for r in rows) / len(rows),
            "n": f"{len(pos)}+{len(neg)}",
        }
    (RES / "scores.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = ["答案正确率", "引用正确率", "误拒率", "拒答率", "格式遵循率", "结构遵循率"]
    print("模型".ljust(18) + "".join(c.ljust(10) for c in cols))
    for name, s in table.items():
        print(name.ljust(20) + "".join(f"{s[c]:.0%}".ljust(12) for c in cols))

if __name__ == "__main__":
    main()
