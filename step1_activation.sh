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

for dataset_name in ${dataset_names[@]}; do 
    for model_name in ${model_names[@]}; do 
        for fixedness in ${fixednesses[@]}; do 
        for semantics in ${semantics[@]}; do 
        for count in ${counts[@]}; do 
            # ------------------------------------------------------------------------------------
            # # 1. Gather Activations 
            # # ------------------------------------------------------------------------------------
            for split in train test; do 
                echo "----------------------------------------"
                echo "Generating ${model_name} for dataset ${dataset_name} split ${split}"
                echo "with fixedness ${fixedness} semantics ${semantics} count ${count}"
                echo "----------------------------------------"
                python store_activations.py \
                    --model_name ${model_name} \
                    --dataset_name ${dataset_name} \
                    --fixedness ${fixedness} \
                    --semantics ${semantics} \
                    --count ${count} \
                    --split ${split} \
                    --batch_size ${llm_batch_size}
            done 
        done 
        done 
        done 
    done 
done 