import random
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def train_prober(prober, train_hiddens, train_labels, test_hiddens, test_labels, 
                  epochs, eval_every, batch_size, lr, verbose=False, **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prober.to(device)
    optimizer = torch.optim.Adam(prober.parameters(), lr=lr)
    train_dataset = TensorDataset(train_hiddens, train_labels)
    test_dataset = TensorDataset(test_hiddens, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    losses = []
    train_accs = []
    test_accs = []
    prober.train()
    eval_steps = []
    if verbose:
        pbar = tqdm(range(epochs))
    else:
        pbar = range(epochs)
        
    best_performance = 0
    best_prober = {k: v.detach().cpu().clone() for k, v in prober.state_dict().items()}
    
    thresholds = [] 

    for epoch in pbar:
        all_outputs = [] 
        train_labels = []
        avg_loss = 0
        for batch in train_dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = prober.forward(X)
            loss = prober.compute_loss(outputs, y)
            
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            all_outputs.append(outputs)
            train_labels.append(y)
            
        all_outputs = torch.cat(all_outputs).reshape(-1)
        train_labels = torch.cat(train_labels).reshape(-1)
        # best threshold
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            threshold, train_acc = best_threshold_accuracy(all_outputs, train_labels)
            prober.threshold = threshold
            thresholds.append(threshold)
            test_acc = evaluate_prober(prober, train_hiddens, train_labels, 
                                                                   test_hiddens, test_labels, device)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            eval_steps.append(epoch + 1)
            avg_loss /= len(train_dataloader)
            losses.append(avg_loss)
            
            if test_acc > best_performance:
                best_performance = test_acc
                best_prober = {k: v.detach().cpu().clone() for k, v in prober.state_dict().items()}
            # pbar.set_postfix(train_acc=train_acc, test_acc=test_acc, threshold=threshold)
            print(f"Epoch {epoch + 1}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Threshold: {threshold:.4f}")
                
    prober.load_state_dict(best_prober)
    return prober, losses, train_accs, test_accs, eval_steps, thresholds

def evaluate_prober(prober, train_hiddens, train_labels, test_hiddens, test_labels, device):
    train_dataset = TensorDataset(train_hiddens, train_labels)
    test_dataset = TensorDataset(test_hiddens, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    is_train = True if train_dataloader else False
    prober.eval()
    with torch.no_grad():
        # find the best threshold with train set
        # train_accs = []
        test_accs = []
        # for batch in train_dataloader:
        #     X, y = batch
        #     X = X.to(device)
        #     y = y.to(device)
        #     train_preds, _ = prober.predict(X, threshold=prober.threshold)
        #     train_acc = (train_preds == y).float()
        #     train_accs.append(train_acc)
            
        for batch in test_dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            test_preds, _ = prober.predict(X, threshold=prober.threshold)
            test_acc = (test_preds == y).float()
            test_accs.append(test_acc)
    # train_accs = torch.cat(train_accs)
    # train_acc = train_accs.mean() 
    test_accs = torch.cat(test_accs)
    test_acc = test_accs.mean()
    
    # train_acc = train_acc.item()
    test_acc = test_acc.item()
    prober.train() if is_train else prober.eval()
    
    # return train_acc, test_acc
    return test_acc
    

def best_threshold_accuracy(y_score: torch.Tensor, y_true: torch.Tensor):
    assert y_score.ndim == 1 and y_true.ndim == 1 and y_score.size(0) == y_true.size(0)

    scores, idx = torch.sort(y_score, descending=True)
    labels = y_true[idx].to(torch.long)

    tp_cum = torch.cumsum(labels, dim=0)
    fp_cum = torch.cumsum(1 - labels, dim=0)
    P = labels.sum()
    N = labels.numel() - P

    tn = N - fp_cum
    tp = tp_cum

    acc = (tp + tn).float() / (P + N)

    best_idx = torch.argmax(acc)
    if best_idx < scores.numel() - 1:
        thr = 0.5 * (scores[best_idx] + scores[best_idx + 1])
    else:
        thr = scores[best_idx]

    return float(thr), float(acc[best_idx])


# ==============================

MODELS = [    
    'meta-llama/Llama-3.3-70B-Instruct',  # https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
    'meta-llama/Llama-3.1-8B-Instruct',  # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    
    'meta-llama/Llama-4-Scout-17B-16E-Instruct',  # https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct',  # https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct
    
    'google/gemma-3-4b-it',   # https://huggingface.co/google/gemma-3-4b-it
    'google/gemma-3-12b-it', # https://huggingface.co/google/gemma-3-12b-pt
    'Qwen/Qwen2.5-7B-Instruct', # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
    'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct', # https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
]

import torch 
import datasets 
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(name:str, device_map="auto"):
    """
    Get model and tokenizer
    Args:
        name: model name
    Returns:
        model and tokenizer
    """
    # For decoder-only models, left padding is crucial for correct generation
    model = AutoModelForCausalLM.from_pretrained(name, device_map=device_map, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(name, device_map=device_map, padding_side="left")
    assert tokenizer.chat_template is not None, "Tokenizer does not have a chat template"
    # Ensure consistent token settings
    tokenizer.pad_token = tokenizer.eos_token
    
    # Double-check padding side is set correctly
    if tokenizer.padding_side != "left":
        print(f"Warning: Tokenizer padding_side was not set to 'left'. Setting it now.")
        tokenizer.padding_side = "left"
    
    return model, tokenizer

def get_llm_block(llm, llm_name):
    if llm_name == "gpt2":
        block = llm.transformer.h
    elif 'meta-llama' in llm_name:
        block = llm.model.layers
    elif 'Qwen' in llm_name:
        block = llm.model.layers
    else:
        raise ValueError(f"Unsupported model: {llm_name}")
    return block


def get_num_layers(llm_name):
    if llm_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        return 32
    elif llm_name == "Qwen/Qwen2.5-7B-Instruct":
        return 28
    else:
        raise ValueError(f"Unsupported model: {llm_name}")

def get_mlp_down_proj(llm_name, block):
    if 'meta-llama' in llm_name:
        module = block.mlp.down_proj
    elif 'Qwen' in llm_name:
        module = block.mlp.down_proj
    else:
        raise ValueError(f"Unsupported model: {llm_name}")
    return module

def get_mlp_up_proj(llm_name, block):
    if 'meta-llama' in llm_name:
        module = block.mlp.up_proj
    elif 'Qwen' in llm_name:
        module = block.mlp.up_proj
    else:
        raise ValueError(f"Unsupported model: {llm_name}")
    return module