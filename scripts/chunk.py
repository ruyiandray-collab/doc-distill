"""把 corpus/text/*.txt 切成段落块 → data/chunks.jsonl

切法：先按"第X条"或换行分成小单元，再把相邻小单元拼到 300~800 字。
法规按条切，能保证一个块里是完整的条款，引用时不会断在半句。
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "corpus" / "text"
OUT = ROOT / "data" / "chunks.jsonl"
MIN, MAX = 300, 800

def unwrap(text: str) -> str:
    """PDF 按版面换行，会把一句话断成好几行：去掉单独的页码行，把不是句末的换行接回去。"""
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)
    lines, out = text.split("\n"), []
    for ln in (l.strip() for l in lines):
        if not ln:
            continue
        new_item = re.match(r"(第[一二三四五六七八九十百零]+[章条节])|（[一二三四五六七八九十]+）|\d+[、.．]|[一二三四五六七八九十]+、", ln)
        if out and not new_item and not re.search(r"[。；;：:！？!?）)]$", out[-1]):
            out[-1] += ln
        else:
            out.append(ln)
    return "\n".join(out)

def units(text: str) -> list[str]:
    text = re.sub(r"\n(?=第[一二三四五六七八九十百零]+条)", "\n\x00", text)   # 条款前打标记
    parts = text.split("\x00") if "\x00" in text else [text]
    out = []
    for p in parts:
        while len(p) > MAX:                       # 超长的条款或表格，按换行再切
            cut = p.rfind("\n", 0, MAX)
            cut = cut if cut > MIN else MAX
            out.append(p[:cut]); p = p[cut:]
        out.append(p)
    return [u.strip() for u in out if u.strip()]

def chunks(text: str) -> list[str]:
    res, buf = [], ""
    for u in units(text):
        if buf and len(buf) + len(u) > MAX:
            res.append(buf); buf = ""
        buf = f"{buf}\n{u}" if buf else u
        if len(buf) >= MIN and re.match(r"第[一二三四五六七八九十百零]+条", u):
            pass                                   # 条款块攒够长度也继续合并，直到超过 MAX
    if buf:
        res.append(buf)
    # 太短的尾块并到前一块
    merged = []
    for c in res:
        if merged and len(c) < 120:
            merged[-1] += "\n" + c
        else:
            merged.append(c)
    return merged

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for p in sorted(TEXT.glob("*.txt")):
            doc = re.sub(r"^\d+_", "", p.stem)
            cs = chunks(unwrap(p.read_text(encoding="utf-8")))
            for i, c in enumerate(cs):
                f.write(json.dumps({"id": f"{p.stem[:2]}-{i:03d}", "doc": doc, "text": c}, ensure_ascii=False) + "\n")
            n += len(cs)
            print(f"{doc}\t{len(cs)} 块\t平均 {sum(map(len, cs)) // len(cs)} 字")
    print("合计", n, "块 →", OUT)
