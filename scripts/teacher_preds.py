"""老师模型 DeepSeek 用完全相同的系统提示和资料回答 raft_test → preds/teacher_deepseek.jsonl
同时记录 token 用量，用来算"每千次调用成本"。"""
import hashlib, asyncio, json, os, time
from pathlib import Path
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
TEST = json.loads((ROOT / "data/raft_test.json").read_text(encoding="utf-8"))
OUT = ROOT / "preds"; OUT.mkdir(exist_ok=True)

async def main():
    client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])
    sem = asyncio.Semaphore(8)
    async def one(x):
        async with sem:
            t = time.time()
            r = await client.chat.completions.create(model="deepseek-chat", temperature=0, max_tokens=512, messages=[
                {"role": "system", "content": x["system"]}, {"role": "user", "content": x["instruction"]}])
            return r.choices[0].message.content, r.usage, time.time() - t
    res = await asyncio.gather(*[one(x) for x in TEST])
    with (OUT / "teacher_deepseek.jsonl").open("w", encoding="utf-8") as f:
        for x, (text, _, _) in zip(TEST, res):
            f.write(json.dumps({"qid": hashlib.md5((x["instruction"] + "\x00" + x["output"]).encode()).hexdigest()[:12], "meta": x["meta"], "ref": x["output"], "pred": text}, ensure_ascii=False) + "\n")
    tin = sum(u.prompt_tokens for _, u, _ in res); tout = sum(u.completion_tokens for _, u, _ in res)
    stat = {"name": "teacher_deepseek", "prompt_tokens": tin, "completion_tokens": tout,
            "avg_latency_s": round(sum(l for *_, l in res) / len(res), 2)}
    (OUT / "teacher_usage.json").write_text(json.dumps(stat, ensure_ascii=False), encoding="utf-8")
    print(stat)

asyncio.run(main())
