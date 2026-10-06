"""把 corpus/raw 里的 PDF / DOCX / HTML 统一转成 corpus/text/*.txt"""
import re, sys
from pathlib import Path
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup

RAW = Path(__file__).resolve().parent.parent / "corpus" / "raw"
OUT = RAW.parent / "text"

def read(p: Path) -> str:
    if p.suffix == ".pdf":
        return "\n".join(pg.extract_text() or "" for pg in PdfReader(p).pages)
    if p.suffix == ".docx" or p.read_bytes()[:2] == b"PK":   # 有些 .doc 其实是 docx
        return "\n".join(par.text for par in docx.Document(p).paragraphs)
    if p.suffix in (".html", ".htm"):
        raw = p.read_bytes()
        html = raw.decode("utf-8", "ignore") if b"utf-8" in raw[:3000].lower() else raw.decode("gb18030", "ignore")
        soup = BeautifulSoup(html, "html.parser")
        node = soup.find(id=re.compile("UCAP-CONTENT|zoom|content", re.I)) or soup.body
        return node.get_text("\n")
    raise ValueError(p)

def clean(t: str) -> str:
    t = re.sub(r"[ \u3000\xa0]+", " ", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", t)   # 去掉中文字间空格
    t = re.sub(r"\n\s*(— ?\d+ ?—|- ?\d+ ?-)\s*\n", "\n", t)          # 去页码
    return re.sub(r"\n{3,}", "\n\n", t).strip()

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for p in sorted(RAW.iterdir()):
        if p.suffix == ".doc" and p.read_bytes()[:2] != b"PK":
            print(f"{p.name}\t旧版 .doc，跳过（文本已用 catdoc 在服务器上转好）")
            continue
        t = clean(read(p))
        (OUT / (p.stem + ".txt")).write_text(t, encoding="utf-8")
        print(f"{p.name}\t{len(t)} chars")
