#!/bin/bash

torchrun --standalone --nproc_per_node=4 train_adapt.py config/train_gpt2_adapt_af_vietnamese.py
torchrun --standalone --nproc_per_node=4 train_adapt.py config/train_gpt2_adapt_af_chinese.py


torchrun --standalone --nproc_per_node=4 train_adapt.py config/train_gpt2_adapt_chinese.py
torchrun --standalone --nproc_per_node=4 train_adapt.py config/train_gpt2_adapt_vietnamese.py
torchrun --standalone --nproc_per_node=4 train_adapt.py config/train_gpt2_adapt_french.py