"""下载资料库原文到 corpus/raw/（原文不放进仓库，用这个脚本从政府网站取）

  python scripts/fetch_corpus.py && python scripts/extract.py && python scripts/chunk.py

注意：部分 .gov.cn 网站不对境外 IP 开放（例如第 04 份），境外网络会下载失败，
可以在境内服务器上下载后拷回来；04 是旧版 .doc，需要用 catdoc 转文本（见 extract.py）。
"""
import subprocess
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "corpus" / "raw"
SOURCES = {
    "01_工贸企业重大事故隐患判定标准.pdf": "https://yjglj.beijing.gov.cn/attach/0/c18dd63fdbc5495ebb5d0fbd321f1dcc.pdf",
    "03_仓库防火安全管理规则.pdf": "https://yw.xfjyzd.sh.gov.cn/uploads/xiaofang/doc/%E4%BB%93%E5%BA%93%E9%98%B2%E7%81%AB%E5%AE%89%E5%85%A8%E7%AE%A1%E7%90%86%E8%A7%84%E5%88%99.pdf",
    "04_租赁厂房和仓库消防安全管理办法.doc": "https://sc.119.gov.cn/scxfjyzd/zcwj/2023/9/26/dcd41fc29783452da91efba2638f2c31/files/%E5%9B%BD%E5%AE%B6%E6%B6%88%E9%98%B2%E6%95%91%E6%8F%B4%E5%B1%80%E5%85%B3%E4%BA%8E%E5%8D%B0%E5%8F%91%E3%80%8A%E7%A7%9F%E8%B5%81%E5%8E%82%E6%88%BF%E5%92%8C%E4%BB%93%E5%BA%93%E6%B6%88%E9%98%B2%E5%AE%89%E5%85%A8%E7%AE%A1%E7%90%86%E5%8A%9E%E6%B3%95%EF%BC%88%E8%AF%95%E8%A1%8C%EF%BC%89%E3%80%8B%E7%9A%84%E9%80%9A%E7%9F%A5(1).doc",
    "05_上海应急局_仓储检查指引.pdf": "https://yjglj.sh.gov.cn/cmsres/78/78b4ef05164c4a56af5eadee9e532e5b/222286927debf483b0143fb970c5a332.pdf",
    "06_上海市场监管局_叉车安全.doc": "https://scjgj.sh.gov.cn/cmsres/31/316fa6ca9f3c4d7d9cd6d9256604b20b/d848d452417b89dc52ed7736130526ef.doc",
    "07_上海应急局_危化品储存.pdf": "https://yjglj.sh.gov.cn/cmsres/f5/f5b6c42ed8f445139addf137cc66d4f4/41ec57d5581db3a1f595d4f1a8f907a2.pdf",
    "08_特种设备安全法.html": "https://www.gov.cn/zhengce/2013-06/30/content_2602288.htm",
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        r = subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0", "--max-time", "120", "-o", str(RAW / name),
                            "-w", "%{http_code} %{size_download}", url], capture_output=True, text=True)
        print(name, r.stdout or r.stderr[:80])
