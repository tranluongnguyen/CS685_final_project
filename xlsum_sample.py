"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from tqdm import tqdm
import datasets
import pickle
import transformers

# -----------------------------------------------------------------------------
model_path = "/content/drive/MyDrive/Cs685/weights/xlsum_en/ckpt_best_xlsum_en_standard.pt"
embed_path = "english"


# model_path = "/content/drive/MyDrive/Cs685/weights/xlsum_en/ckpt_best_xlsum_en_standard.pt"
# embed_path = "english"
# model_path = "/content/drive/MyDrive/Cs685/weights/xlsum_en/ckpt_best_xlsum_en_standard.pt"
# embed_path = "english"

# model_path = "out/lowresource_std_vietnamese/ckpt_latest.pt"
# model_path = "out/standard/ckpt_iter_63000.pt"

init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if os.path.exists(model_path):
    print(f"Loading self-training model from {model_path}")
    ckpt_path = model_path
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])

    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # breakpoint()
    model.load_state_dict(state_dict)
elif init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

if os.path.exists(embed_path):
    print(f"Loading embedding params from {embed_path}")
    embed_checkpoint = torch.load(embed_path, map_location='cpu')
    embed_state_dict = embed_checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(embed_state_dict.items()):
        if k.startswith(unwanted_prefix):
            embed_state_dict[k[len(unwanted_prefix):]] = embed_state_dict.pop(k)
    model.load_state_dict(embed_state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)



if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# NOTE overide the tokenizer if we are using a different language
if "vietnamese" in model_path:
    print("override vietnamese tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained("data/vietnamese/vi_tokenizer")
    encode = lambda s: tokenizer.encode(s)
    decode = lambda l: tokenizer.decode(l)
elif "french" in model_path:
    print("override french tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained("data/french/fr_tokenizer")
    encode = lambda s: tokenizer.encode(s)
    decode = lambda l: tokenizer.decode(l)
elif "chinese" in model_path:
    print("override chinese tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained("data/chinese/zh_tokenizer")
    encode = lambda s: tokenizer.encode(s)
    decode = lambda l: tokenizer.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()


def generate_xlsum(model, text, max_new_tokens):
    with torch.no_grad():
      with ctx:
        start_ids = encode(text)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        idx = x
        ids_answer = []
        for max_token in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            while idx_next[0][0].item() == 198:
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # append sampled index to the running sequence and continue
            # if "vietnamese" or "french" or "chinese" in embed_path:
            #    eos_token = tokenizer.eos_token_id
            # else:
            #    eos_token = enc.eot_token
            eos_token = enc.eot_token
            if idx_next[0][0].item() == eos_token:
              if len(ids_answer) > 1:
                break
              else:
                continue
            else:
              ids_answer.append(idx_next[0][0].item())
            idx = torch.cat((idx, idx_next), dim=1)

        return idx, ids_answer

# def evaluate_xlsum(lang='english', pred_file='xlsum_en_std_predict.txt', gold_file='xlsum_en.pkl'):
#     print("Start eval")
#     ds = datasets.load_dataset('csebuetnlp/xlsum', lang)
#     qa_pairs = []
#     for example in ds['test']:
#         context = "Content: " + example['title'] + " " + example['text'] + " \nSummary: "
#         qa_pairs.append((context, example['text']))
#     predict_answers = []
#     with open(gold_file, 'wb') as f:
#         pickle.dump(qa_pairs, f)
#     error=0



#     with open(pred_file, 'a') as f:
#       for pair in tqdm(qa_pairs):
#         ids_qa, ids_ans = generate_xlsum(model, pair[0], max_new_tokens)
#         ans = decode(ids_ans)
#         ans = ans.replace('\n', ' ').replace('\r', '')
#         if len(ans) < 1:
#             error +=1
#             ans = '\n'
#         if ans[-1] != '\n':
#             ans += '\n'
#         print(ans, '\n------\nstart', decode(ids_qa[0].tolist()))
#         print('---------------')
#         f.write(ans)


def evaluate_xlsum(lang='english', pred_file='xlsum_en_std_predict.txt', gold_file='xlsum_en.pkl'):
    print("Start new eval")
    ds = datasets.load_dataset('csebuetnlp/xlsum', lang)
    qa_pairs = []
    for example in ds['test']:
        context = "Content: " + example['title'] + " " + example['text'] + " \nSummary: "
        qa_pairs.append((context, example['text']))
    predict_answers = []

    eos_token = enc.eot_token
    with open(gold_file, 'wb') as f:
        pickle.dump(qa_pairs, f)
    with open(pred_file, 'a') as f:
        with torch.no_grad():
            with ctx:
                for pair in tqdm(qa_pairs):
                    start_ids = encode(pair[0])
                    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
                    if x.size(1)+max_new_tokens > model.config.block_size:
                        x = x[:, -(model.config.block_size+max_new_tokens):]
                    start_idx = x.size(1)
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    end_idx = y.size(1)
                    for idx in range(y.size(1)):
                        if y[0][idx] == eos_token:
                            end_idx = idx
                    ans = decode(y[0][start_idx:end_idx].tolist())
                    ans = ans.replace('\n', ' ').replace('\r', '')
                    predict_answers.append(ans)
                    print(ans)
                    f.write(ans)
            

evaluate_xlsum(lang='english', pred_file='xlsum_en_std_predict.txt', gold_file='xlsum_en.pkl')
# evaluate()