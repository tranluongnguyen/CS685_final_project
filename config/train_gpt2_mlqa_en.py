# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M-mlqa_en'

dataset = "mlqa_en"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# NOTE overide
learning_rate = 6e-5

# this makes total number of tokens be 300B
warmup_iters = 100
max_iters = 1400
lr_decay_iters = 1400
# reset_interval = 100000

#
save_iter_interval = 200

# eval stuff
eval_interval = 200
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# pretrain_path = "out/standard/ckpt_iter_63000.pt"
# pretrain_path = "out/active_forget/ckpt_iter_87000.pt"
pretrain_path = "out/noise/ckpt_iter_60000.pt"

# CS685_final_project/out/noise/ckpt_iter_60000.pt