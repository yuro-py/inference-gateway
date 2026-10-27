# install required packages
pip install -r requirements.txt

# terminal 1: vLLM from your baseline at port 8000
vllm serve Qwen/Qwen2.5-0.5B-Instruct --gpu-memory-utilization 0.65 --max-model-len 1024 --max-num-seqs 1

# terminal 2: the gateway at port 9000
uvicorn gateway:app --port 9000

# terminal 3: the test
python load_test.py 10
# inference-gateway
