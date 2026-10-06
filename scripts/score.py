"""给预测文件打分 → results/scores.json + 控制台表格

  python scripts/score.py                      # 读 preds/*.jsonl
  python scripts/score.py --preds-dir preds_x  # 读别的目录，结果写 results/scores_preds_x.json
  python scripts/score.py --stamp              # 给旧预测文件补写 qid（只在对齐校验通过后写）

指标（测试集 71 题：有正确原文的 55 题 + 应拒答的 16 题）：
  答案正确率  有原文的题，DeepSeek 当裁判对照参考答案判 正确=1 / 部分正确=0.5 / 错误=0
  引用正确率  有原文的题：每条引文都绑定到它前面标注的资料编号，必须逐字出自那一段；
              且至少有一条引文标注的是正确资料。见 cite_ok
  误拒率      有原文的题，却拒绝回答
  拒答率      没有原文的题，正确地拒绝回答（越高越好，说明不编造）
  格式遵循率  输出里同时有「依据」和「回答」
预测与题目的对应关系用 qid（题目内容的哈希）校验，防止行顺序错位或测试集重新生成后误用旧预测。
裁判结果缓存在 results/judge_cache.json，键里包含题目指纹、参考答案、回答和裁判版本，任何一项变了都会重判。
"""
import argparse, asyncio, hashlib, json, os, re
from pathlib import Path
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TEST = json.loads((ROOT / "data/raft_test.json").read_text(encoding="utf-8"))
QUESTION = [x["instruction"].split("问题：")[-1] for x in TEST]
CACHE_F = RES / "judge_cache.json"

def qid(x):
    """题目指纹：资料 + 问题 + 参考答案。测试集任何一题变了，qid 就对不上。"""
    return hashlib.md5((x["instruction"] + "\x00" + x["output"]).encode()).hexdigest()[:12]
QIDS = [qid(x) for x in TEST]

norm = lambda s: re.sub(r"[\s，。、；：,.;:“”\"'「」『』()（）]", "", s)
REFUSE = re.compile(r"(无法|不能).{0,6}(回答|判断|作答|确定)|并未(涉及|规定|列出|提及)|没有(与该问题|相关|直接相关|提及|涉及)|未(提及|涉及|包含|找到)|不包含")
QUOTE = re.compile(r"[「“\"『]([^」”\"』]{8,})[」”\"』]")

def is_refusal(t):
    """只看「回答」部分的开头：依据部分里出现"没有提及"之类的话不算拒答。"""
    m = list(re.finditer(r"回答[*\s]*[：:]", t))
    ans = t[m[-1].end():] if m else t
    return bool(REFUSE.search(ans[:120]))

def parse_docs(instruction):
    """把输入拆成 {资料编号: 原文}；问题正文不算资料，引用问题里的话不能通过校验。"""
    body = instruction.split("\n\n问题：")[0]
    return {k: t for k, t in re.findall(r"\[资料(\d)\]《[^》]*》\n(.*?)(?=\n\n\[资料\d\]|\Z)", body, re.S)}

def cite_ok(pred, ref, instruction):
    """逐条引文绑定资料编号：
    - 每条引文（≥8 字）取它前面最近一次出现的"资料k"作为来源，必须逐字出自第 k 段；没有标注编号的引文判错；
    - 至少一条引文的来源是正确资料（参考答案里的编号）；
    任何一条引文对不上（编造、张冠李戴、引用问题原文）整题判错。"""
    gold = re.search(r"资料(\d)", ref).group(1)
    docs = parse_docs(instruction)
    head = re.search(r"回答[*\s]*[：:]", pred)
    basis = pred[: head.start()] if head else pred
    labels = [(m.start(), m.group(1)) for m in re.finditer(r"资料\s*(\d)", basis)]
    hit_gold = False
    for m in QUOTE.finditer(basis):
        src = [k for pos, k in labels if pos < m.start()]
        if not src or src[-1] not in docs or norm(m.group(1)) not in norm(docs[src[-1]]):
            return False
        hit_gold |= src[-1] == gold
    return hit_gold

def load_runs(pred_dir, stamp):
    runs = {}
    for p in sorted(pred_dir.glob("*.jsonl")):
        if p.stem.startswith("speed"):          # speed.jsonl / speed_v2.jsonl 是测速记录，不是预测
            continue
        rows = [json.loads(l) for l in p.open(encoding="utf-8")]
        if len(rows) != len(TEST) or not all({"meta", "ref", "pred"} <= r.keys() for r in rows):
            raise SystemExit(f"{p.name}：不是完整的预测文件（应为 {len(TEST)} 条，每条含 meta/ref/pred）")
        legacy = False
        for i, (r, x) in enumerate(zip(rows, TEST)):
            if "qid" in r:
                ok = r["qid"] == QIDS[i]
            else:   # 旧文件没有 qid：用 meta + 参考答案逐行核对
                legacy, ok = True, r["meta"] == x["meta"] and r["ref"] == x["output"]
            if not ok:
                raise SystemExit(f"{p.name} 第 {i + 1} 行与测试集第 {i + 1} 题对不上（顺序错位，或测试集已重新生成）")
        if legacy:
            if stamp:
                with p.open("w", encoding="utf-8") as f:
                    for r, q in zip(rows, QIDS):
                        f.write(json.dumps({"qid": q, **r}, ensure_ascii=False) + "\n")
                print(f"{p.name}：已补写 qid")
            else:
                print(f"注意：{p.name} 没有 qid，已用 meta+参考答案核对通过（--stamp 可补写）")
        runs[p.stem] = rows
    return runs

JUDGE = """你是严格的阅卷老师。
问题：{q}
参考答案：{ref}
学生回答：{pred}
1. refused：学生是否在拒答，即最终没有给出实质结论，而是表示资料不足、无法回答？（只在依据里说"某段资料没提到"、但最后给出了结论的，不算拒答）
2. verdict：对照参考答案，结论与关键数值/条件一致为 correct；结论对但遗漏重要条件或有小错为 partial；结论错误、答非所问或拒答为 wrong。
只输出 JSON：{{"refused": true 或 false, "verdict": "correct|partial|wrong"}}"""
NO_ANSWER_REF = "（本题提供的资料中没有答案，正确做法是拒答）"
JUDGE_MODEL = "deepseek-chat"
# 裁判提示词或裁判模型一改，指纹就变，旧判定自动失效
JUDGE_VER = hashlib.md5((JUDGE + JUDGE_MODEL).encode()).hexdigest()[:8]

def judge_key(i, ref_ans, pred):
    """缓存键 = 题目指纹（含资料与参考答案）+ 本次送给裁判的参考答案 + 学生回答 + 裁判版本"""
    return hashlib.md5("\x00".join([QIDS[i], ref_ans, pred, JUDGE_VER]).encode()).hexdigest()

async def judge_all(items, cache):
    client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])
    sem = asyncio.Semaphore(10)
    async def one(key, q, ref, pred):
        if key in cache: return
        async with sem:
            r = await client.chat.completions.create(model=JUDGE_MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": JUDGE.format(q=q, ref=ref, pred=pred)}])
            cache[key] = json.loads(r.choices[0].message.content)
    await asyncio.gather(*[one(*it) for it in items])

def main(a):
    RES.mkdir(exist_ok=True)
    pred_dir = ROOT / a.preds_dir
    cache = json.loads(CACHE_F.read_text(encoding="utf-8")) if CACHE_F.exists() else {}
    runs = load_runs(pred_dir, a.stamp)
    todo = []
    for name, rows in runs.items():
        for i, r in enumerate(rows):
            ref_ans = r["ref"].split("回答：")[-1] if r["meta"]["has_oracle"] else NO_ANSWER_REF
            key = judge_key(i, ref_ans, r["pred"])
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
            "引用正确率": sum(cite_ok(r["pred"], r["ref"], TEST[i]["instruction"])
                          for i, r in enumerate(rows) if r["meta"]["has_oracle"]) / len(pos),
            "误拒率": sum(refused(r) for r in pos) / len(pos),
            "拒答率": sum(refused(r) for r in neg) / len(neg),
            "拒答率_正则": sum(is_refusal(r["pred"]) for r in neg) / len(neg),
            # 严格：开头有「依据」二字；结构：有资料编号引用（或拒答）且有「回答」——RAFT 模型常漏写「依据：」这两个字
            "格式遵循率": sum("依据" in r["pred"] and "回答" in r["pred"] for r in rows) / len(rows),
            "结构遵循率": sum(("资料" in r["pred"].split("回答")[0] or refused(r)) and "回答" in r["pred"] for r in rows) / len(rows),
            "n": f"{len(pos)}+{len(neg)}",
        }
    out = RES / ("scores.json" if a.preds_dir == "preds" else f"scores_{Path(a.preds_dir).name}.json")
    out.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = ["答案正确率", "引用正确率", "误拒率", "拒答率", "格式遵循率", "结构遵循率"]
    print("模型".ljust(18) + "".join(c.ljust(10) for c in cols))
    for name, s in table.items():
        print(name.ljust(20) + "".join(f"{s[c]:.0%}".ljust(12) for c in cols))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default="preds")
    ap.add_argument("--stamp", action="store_true", help="给通过核对的旧预测文件补写 qid")
    main(ap.parse_args())
