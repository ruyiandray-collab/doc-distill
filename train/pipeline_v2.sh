#!/bin/bash
# v2：先做 token 对齐检查（不通过就停），再训练三个模型，最后评测（只测 LoRA 模型）
W=/root/autodl-tmp/work; L=/root/autodl-tmp/logs
cd $W
source /root/venvs/lf/bin/activate
python scripts/check_template.py > $L/v2_check.log 2>&1
grep -q "v2 qwen3 + enable_thinking=false: prompt 与推理一致 = True" $L/v2_check.log || { echo fail > $L/v2.failed; exit 1; }
for cfg in raft_8b raft_1.7b plain_8b; do
  [ -f $L/v2_$cfg.done ] && continue
  llamafactory-cli train train/v2/$cfg.yaml > $L/v2_$cfg.log 2>&1
  echo $? > $L/v2_$cfg.done
done
deactivate; source /root/venvs/vllm/bin/activate
export CKPT_DIR=/root/autodl-tmp/ckpt/v2 PRED_DIR=$W/preds_v2 SKIP_BASE=1
python scripts/eval_gen.py 8b   > $L/v2_eval_8b.log 2>&1
python scripts/eval_gen.py 1.7b > $L/v2_eval_1.7b.log 2>&1
echo done > $L/v2_eval.done
