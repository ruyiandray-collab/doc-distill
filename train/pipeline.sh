#!/bin/bash
# 一键：后台装 vLLM 的同时训练；训练成功且 vLLM 安装成功后才跑评测
cd /root/autodl-tmp/work
L=/root/autodl-tmp/logs
rm -f $L/vllm_setup.done
nohup bash train/setup_vllm.sh > $L/vllm_setup.log 2>&1 &
bash train/run_train.sh || { echo "训练失败，见 $L/*.failed"; exit 1; }
while [ ! -f $L/vllm_setup.done ]; do sleep 10; done
[ "$(cat $L/vllm_setup.done)" = "0" ] || { echo "vLLM 安装失败，见 $L/vllm_setup.log"; exit 1; }
bash train/run_eval.sh
