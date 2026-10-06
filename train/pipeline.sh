#!/bin/bash
# 一键：后台装 vLLM 的同时训练；训练和安装都完成后跑评测
cd /root/autodl-tmp/work
nohup bash train/setup_vllm.sh > /root/autodl-tmp/logs/vllm_setup.log 2>&1 &
bash train/run_train.sh
while [ ! -f /root/autodl-tmp/logs/vllm_setup.done ]; do sleep 10; done
bash train/run_eval.sh
