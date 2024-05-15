# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M-adapt-chinese'

dataset = "chinese"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# NOTE overide
learning_rate = 6e-4

# this makes total number of tokens be 300B
warmup_iters = 500
max_iters = 20000
lr_decay_iters = 20000
# reset_interval = 100000

#
save_iter_interval = 2000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

pretrain_path = "out/active_forget/ckpt_iter_87000.pt"
