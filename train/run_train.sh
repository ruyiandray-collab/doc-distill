#!/bin/bash
# 依次训练三个模型；每个结束写一个标记文件，方便本地轮询
source /root/venvs/lf/bin/activate
cd /root/autodl-tmp/work
for cfg in raft_8b raft_1.7b plain_8b; do
  [ -f /root/autodl-tmp/logs/$cfg.done ] && continue
  llamafactory-cli train train/$cfg.yaml > /root/autodl-tmp/logs/$cfg.log 2>&1
  echo $? > /root/autodl-tmp/logs/$cfg.done
done
