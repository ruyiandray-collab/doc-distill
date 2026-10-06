#!/bin/bash
# v2：先做 token 对齐检查（不通过就停），再训练三个模型，最后评测（只测 LoRA 模型）
# 只有训练成功（退出码 0）才写 .done；失败写 .failed 并停止，修好后重跑会从失败的那个模型继续
W=/root/autodl-tmp/work; L=/root/autodl-tmp/logs
cd $W
source /root/venvs/lf/bin/activate
python scripts/check_template.py > $L/v2_check.log 2>&1
grep -q "v2 qwen3 + enable_thinking=false: prompt 与推理一致 = True" $L/v2_check.log || { echo fail > $L/v2.failed; exit 1; }
for cfg in raft_8b raft_1.7b plain_8b; do
  [ -f $L/v2_$cfg.done ] && continue
  rm -f $L/v2_$cfg.failed
  if llamafactory-cli train train/v2/$cfg.yaml > $L/v2_$cfg.log 2>&1; then
    echo 0 > $L/v2_$cfg.done
  else
    echo $? > $L/v2_$cfg.failed; exit 1
  fi
done
deactivate; source /root/venvs/vllm/bin/activate
export CKPT_DIR=/root/autodl-tmp/ckpt/v2 PRED_DIR=$W/preds_v2 SKIP_BASE=1
python scripts/eval_gen.py 8b   > $L/v2_eval_8b.log 2>&1   || { echo 8b > $L/v2_eval.failed; exit 1; }
python scripts/eval_gen.py 1.7b > $L/v2_eval_1.7b.log 2>&1 || { echo 1.7b > $L/v2_eval.failed; exit 1; }
echo done > $L/v2_eval.done
