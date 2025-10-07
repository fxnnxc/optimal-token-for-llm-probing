import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# Will be Removed Later
# =====================================================================

import os 
import json 
import torch 

from utils import  get_num_layers
from probers import get_prober
from data import get_dataset
from utils import train_prober 
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import set_seed

set_seed(42)

def main(args):
    prober_name = args.prober_name
    model_name = args.model_name
    dataset_name = args.dataset_name
    fixedness = args.fixedness
    semantics = args.semantics
    count = args.count
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    eval_every = args.eval_every

    train_Y_original = torch.load(f"outputs/activations/{model_name}/{dataset_name}/{fixedness}_{semantics}_{count}/train/labels.pt")
    test_Y_original = torch.load(f"outputs/activations/{model_name}/{dataset_name}/{fixedness}_{semantics}_{count}/test/labels.pt")

    config =  OmegaConf.create(vars(args))
    config.save_dir = f"outputs/value_probers/{model_name}/{dataset_name}/{fixedness}_{semantics}_{count}/{prober_name}/{args.init_seed}"
    os.makedirs(config.save_dir, exist_ok=True)
    OmegaConf.save(config, f"{config.save_dir}/config.yaml")

    # ------------------------------------------------------------------------------------
    # Load Activations 
    # ------------------------------------------------------------------------------------
    llm_num_layers = get_num_layers(model_name)
    args.layer_indices = list(range(llm_num_layers))
    center = len(args.layer_indices)//2
    args.layer_indices = args.layer_indices[center-1:center+1]
    pbar = tqdm(args.layer_indices, desc=f"{config.save_dir}")
    for target_layer in pbar:
        train_X = torch.load(f"outputs/activations/{model_name}/{dataset_name}/{fixedness}_{semantics}_{count}/train/layer_{target_layer}_value.pt")
        test_X = torch.load(f"outputs/activations/{model_name}/{dataset_name}/{fixedness}_{semantics}_{count}/test/layer_{target_layer}_value.pt")

        train_X = train_X.float()
        test_X = test_X.float()
    # args.save_dir = os.path.join(
    #     f'outputs/activations/',
    #     f'{args.model_name}/{args.dataset_name}',
    #     f'{args.fixedness}_{args.semantics}_{args.count}/{args.split}',
    # )

        # Use Last Three Tokens as Input 
        if train_X.ndim == 3:
            N1, T1, D1 = train_X.shape
            N2, T2, D2 = test_X.shape
            train_X = train_X.reshape(N1*T1, D1).float()
            test_X = test_X.reshape(N2*T2, D2).float()
            train_Y = train_Y_original.repeat_interleave(T1)
            test_Y = test_Y_original.repeat_interleave(T2)    
            hidden_dim= D1
        else:
            hidden_dim= train_X.shape[1]
            train_Y = train_Y_original
            test_Y = test_Y_original
        # Make Prober
        
        value_prober  = get_prober(prober_name, hidden_dim, num_layers=1, init_seed=args.init_seed)

        prober, losses, train_accs, test_accs, eval_steps, thresholds = train_prober(
                value_prober, 
                train_X, 
                train_Y, 
                test_X, 
                test_Y, 
                epochs=epochs, 
                eval_every=eval_every, 
                batch_size=batch_size, 
                lr=lr,
                verbose=False,
        )
        torch.save(prober, f"{config.save_dir}/layer_{target_layer}_prober.pt") 
        results = {}
        results["losses"] = losses
        results["train_accs"] = train_accs
        results["test_accs"] = test_accs
        results["eval_steps"] = eval_steps
        results["thresholds"] = thresholds
        json.dump(results, open(f"{config.save_dir}/layer_{target_layer}_results.json", "w"))
        
        fig, axes = plt.subplots(1,2, figsize=(7, 3))
        axes[0].plot(losses)
        axes[1].plot(train_accs)
        axes[1].plot(test_accs)
        axes[0].set_title("Train Loss")
        axes[1].set_title("Accuracy")
        axes[1].legend(["Train", "Test"])
        fig.suptitle("Value Prober Results")
        plt.tight_layout()
        plt.savefig(f"{config.save_dir}/layer_{target_layer}_results.png")
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prober_name", type=str, required=True)
    parser.add_argument("--fixedness", type=str, required=True)
    parser.add_argument("--semantics", type=str, required=True)
    parser.add_argument("--count", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--eval_every", type=int, required=True)
    parser.add_argument("--init_seed", type=int, required=True)
    args = parser.parse_args()
    main(args)




