# MiniCPM-Llama3: a GPT-4V Level Multimodal LLM

> https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5

## Files

Inference using Huggingface transformers on NVIDIA GPUs

## Text Generation Inference (TGI) toolkit

## Monitoring Local GPU Usage

```
pip3 install --upgrade nvitop
```

## Requirements
- python 3.10
```
Pillow==10.1.0
torch==2.1.2
torchvision==0.16.2
transformers==4.40.0
sentencepiece==0.1.99
fastapi
tgi
```

## Docker Image

```
docker build -t minicpm .
docker run --gpus all -p 8000:8000 minicpm
```