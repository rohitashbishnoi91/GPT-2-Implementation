# GPT-2-Implementation

A self-learning project implementing the GPT-2 architecture from scratch in PyTorch, trained on the Tiny Shakespeare dataset.

Features
Full GPT-2 model built from the ground up

Supports Single GPU, Distributed Data Parallel (DDP), and Fully Sharded Data Parallel (FSDP) training

Lightweight dataset for quick iteration and experimentation

Usage
Clone the repo

Install dependencies (requirements.txt)

Run training:

bash
Copy
Edit
python train.py --mode [single|ddp|fsdp]
Dataset
Tiny Shakespeare
