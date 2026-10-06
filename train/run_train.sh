#!/bin/bash
# v1：依次训练三个模型。只有成功才写 .done（重跑时跳过）；失败写 .failed 并以非零退出，后续步骤不会继续
source /root/venvs/lf/bin/activate
cd /root/autodl-tmp/work
for cfg in raft_8b raft_1.7b plain_8b; do
  [ -f /root/autodl-tmp/logs/$cfg.done ] && continue
  rm -f /root/autodl-tmp/logs/$cfg.failed
  if llamafactory-cli train train/$cfg.yaml > /root/autodl-tmp/logs/$cfg.log 2>&1; then
    echo 0 > /root/autodl-tmp/logs/$cfg.done
  else
    echo $? > /root/autodl-tmp/logs/$cfg.failed; exit 1
  fi
done
