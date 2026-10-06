"""老师模型（DeepSeek）读每个块，出题 + 给带原文引用的标准答案 → data/qa_raw.jsonl

用法：
  python scripts/gen_qa.py --sample 3     # 先抽 3 块看质量
  python scripts/gen_qa.py                # 全量（已生成的块会跳过，可断点续跑）

质量闸门：quote 必须是原文的逐字片段（去掉空白后做子串匹配），否则丢弃这道题。
"""
import argparse, asyncio, json, os, random, re, time
from pathlib import Path
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
OUT = ROOT / "data" / "qa_raw.jsonl"
MODEL = os.getenv("TEACHER_MODEL", "deepseek-chat")

PROMPT = """你是安全生产法规培训的出题老师。下面是一段法规/指引原文，来自《{doc}》。

请基于这段原文出 {n} 道题，题型尽量不同：
- 事实型：直接问某条规定的具体要求、数值、期限、处罚等；
- 条件判断型：给出一个具体情形，问"是否符合规定 / 是否属于重大事故隐患"；
- 情景应用型：以仓库管理员、安全员、叉车司机等身份提出实际工作中的问题。

要求：
1. 答案必须完全能由这段原文推出，不得使用原文以外的知识；
2. 问题要像真实用户会问的话，不要出现"根据原文""本段"这类字眼，也不要直接照抄条款原句当问题；
3. quote 必须从原文中**逐字复制**一段最关键的依据（20~150 字），不能改写、不能拼接不相邻的句子；
4. answer 用 1~3 句话直接回答，结论在前；结论的定性不能比原文更重，例如原文只说"应当……"，就不能回答成"属于重大事故隐患"；
5. 原文如果是表格转出的文字（检查事项、防范措施、依据挤在一起），quote 只取其中一个完整的单元，不要跨单元拼接。
6. 如果原文只是目录、表头或没有实质规定，返回空列表。

只输出 JSON：{{"items": [{{"type": "事实型|条件判断型|情景应用型", "question": "...", "quote": "...", "answer": "..."}}]}}

原文：
<<<
{text}
>>>"""

norm = lambda s: re.sub(r"\s+", "", s)

async def gen_one(client, chunk, n, sem):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, temperature=0.7,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": PROMPT.format(doc=chunk["doc"], n=n, text=chunk["text"])}])
                items = json.loads(r.choices[0].message.content).get("items", [])
                usage = r.usage
                break
            except Exception as e:
                if attempt == 2:
                    print("失败", chunk["id"], e); return [], None
                await asyncio.sleep(2 ** attempt)
    kept = []
    for it in items:
        ok = bool(it.get("quote")) and norm(it["quote"]) in norm(chunk["text"])
        it.update(chunk_id=chunk["id"], doc=chunk["doc"], quote_ok=ok)
        kept.append(it)
    return kept, usage

async def main(a):
    client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    if a.sample:
        random.seed(7); chunks = random.sample(chunks, a.sample); out = OUT.with_name("qa_sample.jsonl"); mode = "w"
    else:
        done = {json.loads(l)["chunk_id"] for l in OUT.open(encoding="utf-8")} if OUT.exists() else set()
        chunks = [c for c in chunks if c["id"] not in done]; out = OUT; mode = "a"
    sem, t0 = asyncio.Semaphore(a.workers), time.time()
    res = await asyncio.gather(*[gen_one(client, c, a.n, sem) for c in chunks])
    tin = tout = 0; total = good = 0
    with out.open(mode, encoding="utf-8") as f:
        for items, u in res:
            if u: tin += u.prompt_tokens; tout += u.completion_tokens
            for it in items:
                total += 1; good += it["quote_ok"]
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} 块 → {total} 题，引用校验通过 {good}（{good / max(total, 1):.0%}）"
          f" | 输入 {tin} / 输出 {tout} tokens | {time.time() - t0:.0f}s → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--n", type=int, default=3, help="每块出几题")
    ap.add_argument("--workers", type=int, default=8)
    asyncio.run(main(ap.parse_args()))
