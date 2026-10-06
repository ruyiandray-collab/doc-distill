#!/bin/bash
# 等训练全部结束后，分两个进程评测（每个进程结束才能完全释放显存）
source /root/venvs/vllm/bin/activate
cd /root/autodl-tmp/work
rm -f preds/speed.jsonl
python scripts/eval_gen.py 8b   > /root/autodl-tmp/logs/eval_8b.log 2>&1
python scripts/eval_gen.py 1.7b > /root/autodl-tmp/logs/eval_1.7b.log 2>&1
echo done > /root/autodl-tmp/logs/eval.done
