#!/bin/bash
# vLLM 单独一个虚拟环境，和 LLaMA-Factory 的依赖互不干扰
trap 'echo $? > /root/autodl-tmp/logs/vllm_setup.done' EXIT
set -e
/root/miniconda3/bin/uv venv /root/venvs/vllm --python /root/miniconda3/bin/python --allow-existing
source /root/venvs/vllm/bin/activate
/root/miniconda3/bin/uv pip install --index-url https://mirrors.aliyun.com/pypi/simple/ vllm
python -c "import vllm, torch; print('vllm', vllm.__version__, torch.__version__, torch.cuda.is_available())"
