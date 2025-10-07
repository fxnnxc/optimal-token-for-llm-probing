#! /usr/bin/bash

export OMP_NUM_THREADS=8
# ------------------------------------------------------------------------------------
# LLM 
# ------------------------------------------------------------------------------------
model_names=(
    'meta-llama/Meta-Llama-3.1-8B-Instruct'
    'Qwen/Qwen2.5-7B-Instruct'
)
dataset_names=(
    'imdb-small'
    'paradetox'
)
llm_batch_size=16

fixednesses=(
    'fixed'
    'variable'
)
semantics=(
    'syntactical'
    'special'
    'random'
)
counts=(
    'single'
    'multi'
)

prober_batch_size=256
lr=0.001
epochs=100
eval_every=10
prober_name="BCEProber"
init_seed=42
for dataset_name in ${dataset_names[@]}; do 
for model_name in ${model_names[@]}; do 
        for fixedness in ${fixednesses[@]}; do 
        for semantics in ${semantics[@]}; do 
        for count in ${counts[@]}; do 
            # ------------------------------------------------------------------------------------
            # # 1. Gather Activations 
            # # ------------------------------------------------------------------------------------
            echo "----------------------------------------"
            echo "Training Value Prober for ${model_name} with prompt version ${prompt_version} for dataset ${dataset_name}"
            echo "----------------------------------------"
            python train_value_prober.py \
                --model_name ${model_name} \
                --fixedness ${fixedness} \
                --semantics ${semantics} \
                --count ${count} \
                --dataset_name ${dataset_name} \
                --batch_size ${prober_batch_size} \
                --lr ${lr} \
                --epochs ${epochs} \
                --eval_every ${eval_every} \
                --prober_name ${prober_name} \
                --init_seed ${init_seed}
        done 
        done 
        done 
done 
done 