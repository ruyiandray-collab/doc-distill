#!/bin/bash
# 分两个进程评测（每个进程结束才能完全释放显存）；任何一步失败都不写完成标记
source /root/venvs/vllm/bin/activate
cd /root/autodl-tmp/work
L=/root/autodl-tmp/logs
rm -f preds/speed.jsonl $L/eval.done $L/eval.failed
python scripts/eval_gen.py 8b   > $L/eval_8b.log 2>&1   || { echo 8b > $L/eval.failed; exit 1; }
python scripts/eval_gen.py 1.7b > $L/eval_1.7b.log 2>&1 || { echo 1.7b > $L/eval.failed; exit 1; }
echo done > $L/eval.done
