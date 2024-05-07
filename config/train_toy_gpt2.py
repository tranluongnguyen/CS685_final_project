# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

out_dir = 'out-toy-openwebtext'

wandb_log = True
wandb_project = 'toy-owt'
wandb_run_name='toy-gpt2-124M'


dataset = 'toy-openwebtext'



batch_size = 8
block_size = 1024
gradient_accumulation_steps = 1

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 500
lr_decay_iters = 500 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # not super necessary potentially
reset_interval = 10

# eval stuff
eval_interval = 250
eval_iters = 200
log_interval = 100
always_save_checkpoint = False

# weight decay
weight_decay = 1e-1
