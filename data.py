from datasets import load_dataset
from typing import List
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import datasets
import os 


def get_dataset(name:str, split:str, seed=42, data_dir='data_dir'):
    """
    Get dataset
    Args:
        name: name of the dataset
        split: split of the dataset
    Returns:
        dataset
    """
    assert split in ["train", "test"], "We only support train and test split for test dataset"
    
    if name == "sst2-small":
        if split == "test":
            split = "validation"
        dataset = load_dataset("stanfordnlp/sst2", split=split)
        dataset = dataset.shuffle(seed=seed).select([i for i in list(range(872))])   # less than or equal to 872
        dataset = dataset.map(lambda x: {"answer": {0: "negative", 1: "positive"}[x["label"]]})
        dataset = dataset.map(lambda x, i: {"sample_id": i}, with_indices=True)
        dataset = dataset.map(lambda x: {"label": convert_label_to_binary(name, x["answer"])})
        return dataset
    
    elif name == "imdb-small":    
        dataset = datasets.load_dataset("imdb", split=split)
        dataset = dataset.shuffle(seed=seed).select([i for i in list(range(5000))]) 
        dataset = dataset.map(lambda x: {"answer": {0: "negative", 1: "positive"}[x["label"]]})
        dataset = dataset.map(lambda x, i: {"sample_id": i}, with_indices=True)
        dataset = dataset.map(lambda x: {"label": convert_label_to_binary(name, x["answer"])})
        return dataset
    
    
    elif name == "paradetox":
        dataset = datasets.load_dataset("s-nlp/paradetox")
        dataset = dataset['train']
        states = np.random.get_state()
        dataset = dataset.shuffle(seed=seed).select(range(5000))
        
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))
        train_indices = indices[:int(len(dataset)*0.5)]
        test_indices = indices[int(len(dataset)*0.5):]
        np.random.set_state(states)
        
        dataset = dataset.select(train_indices) if split == "train" else dataset.select(test_indices)
        
        new_ds = {
            'text': [],
            'label': [],  # 0: toxic, 1: neutral
            'sample_id': [],
        }
        
        # Process all examples
        for i in tqdm(range(len(dataset)), desc="Processing ParaDetox dataset"):
            toxic_text = dataset[i]['en_toxic_comment']
            neutral_text = dataset[i]['en_neutral_comment']
            
            # Add toxic example
            new_ds['text'].append(toxic_text)
            new_ds['label'].append(1)  # Toxic
            new_ds['sample_id'].append(i)
            
            # Add neutral example
            new_ds['text'].append(neutral_text)
            new_ds['label'].append(0)  # Neutral
            new_ds['sample_id'].append(i)
        
        # Create dataset and split
        dataset = datasets.Dataset.from_dict(new_ds)
        dataset = dataset.map(lambda x: {"answer": {0: "Neutral", 1: "Toxic"}[x["label"]]})
        dataset = dataset.map(lambda x, i: {"sample_id": i}, with_indices=True)
        dataset = dataset.map(lambda x: {"label": convert_label_to_binary(name, x["answer"])})
        return dataset

    elif name == "truthfulqa":
        ds = datasets.load_dataset("truthfulqa/truthful_qa", "generation")

        # Label0: Incorrect
        # Label1: Correct
        
        new_ds = {
            'text':[],
            'answer':[],
            'sample_id':[],
        }
        labels = [
            'incorrect', 
            'correct'
        ]
        ds = ds['validation']
        
        state = np.random.get_state()
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(len(ds))
        train_indices = shuffle_indices[:int(len(ds)*0.5)]
        test_indices = shuffle_indices[int(len(ds)*0.5):]
        np.random.set_state(state)
        
        for label_idx, label in enumerate(labels):
            for i in range(len(ds)):
                if split == "train" and i in train_indices:
                    pass
                elif split == "test" and i in test_indices:
                    pass
                else:
                    continue
                for text in ds[i][f'{label}_answers']:
                    new_ds['text'].append(ds[i]['question'] + "\n" + "Answer:" + text)
                    new_ds['answer'].append(label.capitalize())
                    new_ds['sample_id'].append(i)
                    
        dataset = datasets.Dataset.from_dict(new_ds) 
        dataset = dataset.map(lambda x: {"label": convert_label_to_binary(name, x["answer"])})
        return dataset
    
    elif name == "gsm8k":
        dataset = datasets.load_dataset("openai/gsm8k", "main")

        def extract_answer(answer): return answer.split("####")[1].strip()
        dataset = dataset.map(lambda x: {"answer": extract_answer(x["answer"])})
        dataset = dataset.map(lambda x: {"text": x["question"]})
        dataset = dataset.map(lambda x, i: {"sample_id": i}, with_indices=True)
        dataset = dataset.map(lambda x: {"label": convert_label_to_binary(name, x["answer"])})
        return dataset[split]
    
    elif name == "deontology":
        dataset = pd.read_csv(f"{data_dir}/ethics/deontology/deontology_{split}.csv")
        data = {
            'scenario':[],
            'excuse':[],
            'answer':[],
            'sample_id':[]
        }
        for i in range(len(dataset)):
            data['scenario'].append(dataset.iloc[i]['scenario'])
            data['excuse'].append(dataset.iloc[i]['excuse'])
            data['answer'].append(dataset.iloc[i]['label'])
            data['sample_id'].append(i)
        dataset = datasets.Dataset.from_dict(data)
        dataset = dataset.shuffle(seed=seed).select(range(2500))
        dataset = dataset.map(lambda x: {"label": convert_label_to_binary(name, x["answer"])})
        return dataset

    elif name in ["justice", "virtue"]:
        dataset = pd.read_csv(f"{data_dir}/ethics/{name}/{name}_{split}.csv")
        data = {
            'scenario':[],
            'answer':[],
            'sample_id':[]
        }
        for i in range(len(dataset)):
            if name == "virtue":
                scenario = dataset.iloc[i]['scenario']
                scenario = scenario.replace("[SEP]", "Character trait:")
            else:
                scenario = dataset.iloc[i]['scenario']
            label = dataset.iloc[i]['label']
            data['scenario'].append(scenario)
            data['answer'].append(label)
            data['sample_id'].append(i)
        dataset = datasets.Dataset.from_dict(data)
        dataset = dataset.shuffle(seed=seed).select(range(2500))
        dataset = dataset.map(lambda x: {"label": convert_label_to_binary(name, x["answer"])})
        return dataset
    
    elif "countries" in name:
        countries = pd.read_csv(f"{data_dir}/countries.csv")
        task = name.split("_")[1]
        if task == "capital":
            column = 'capital'
        elif task == "language":
            column = 'languages'
        elif task == "region":
            column = 'region'
        elif task == "callingcodes":
            column = 'callingCodes'
        elif task == "borders":
            column = 'borders'
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        state = np.random.get_state()
        np.random.seed(seed)
        random_indices = np.random.permutation(len(countries))
        train_indices = random_indices[:int(len(countries)*0.7)]
        test_indices = random_indices[int(len(countries)*0.7):]
        np.random.set_state(state)
        
        samples = {
            'country':[],
            'answer':[],
            'sample_id':[]
        }
        for index, (country, item) in enumerate(zip(countries['name.common'].tolist(), countries[column].tolist())):
            if pd.isna(item):
                continue
            if split == "train" and index in train_indices:
                pass
            elif split == "test" and index in test_indices:
                pass
            else:
                continue
            item = item.split(',')[0]
            samples['country'].append(country)
            samples['answer'].append(item)
            samples['sample_id'].append(index)
    
        dataset = datasets.Dataset.from_dict(samples)
        return dataset
    elif name == "logicqa":
        dataset = datasets.load_dataset("lucasmccabe/logiqa")
        dataset = dataset[split]
        data = {
            'question':[],
            'answer':[],
            'sample_id':[]
        }
        for i in range(len(dataset)):
            context = dataset[i]['context']
            query = dataset[i]['query']
            options = dataset[i]['options']
            answer = dataset[i]['correct_option']
            options = "\n".join([f"{j}: {option}" for j, option in enumerate(options)])
            question = f"Context: {context}\nQuery: {query}\nOptions:\n{options}"
            
            data['question'].append(question)
            data['answer'].append(answer)
            data['sample_id'].append(i)
    
        return datasets.Dataset.from_dict(data)
    
    elif name == "popqa":
        dataset = datasets.load_dataset("akariasai/PopQA")
        dataset = dataset['test']
        
        state = np.random.get_state()
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))
        train_indices = indices[:int(len(dataset)*0.7)]
        test_indices = indices[int(len(dataset)*0.7):]
        np.random.set_state(state)
        
        data = {
            'question':[],
            'answer':[],
            'sample_id':[]
        }
        for i in range(len(dataset)):
            if split == "train" and i in train_indices:
                pass
            elif split == "test" and i in test_indices:
                pass
            else:
                continue
            question = dataset[i]['question']
            answer = dataset[i]['possible_answers']
            data['question'].append(question)
            data['answer'].append(answer)
            data['sample_id'].append(i)
        dataset = datasets.Dataset.from_dict(data)
        return dataset
    elif name == "mmlu":
        subsets =  ['anatomy', 'business_ethics', 'clinical_knowledge', 'college_chemistry', 
                    'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'econometrics', 
                    'electrical_engineering', 'formal_logic', 'global_facts', 'high_school_chemistry', 'high_school_mathematics', 
                    'high_school_physics', 'high_school_statistics', 'human_aging', 'logical_fallacies', 'machine_learning', 'miscellaneous', 
                    'philosophy', 'professional_accounting', 'public_relations', 'virology', 'conceptual_physics', 'high_school_us_history', 
                    'astronomy', 'high_school_geography', 'high_school_macroeconomics', 'professional_law']
        data = {
            'question':[],
            'answer':[],
            'sample_id':[],
            'cateogry':[]
        }
        state = np.random.get_state()
        np.random.seed(seed)
        indices = np.random.permutation([i for i in range(100)])
        train_indices = indices[:int(len(indices)*0.7)]
        test_indices = indices[int(len(indices)*0.7):]
        
        train_positive_indices = train_indices[:int(len(train_indices)*0.5)]
        train_negative_indices = train_indices[int(len(train_indices)*0.5):]
        test_positive_indices = test_indices[:int(len(test_indices)*0.5)]
        test_negative_indices = test_indices[int(len(test_indices)*0.5):]
        
        if split == "train":
            positive_indices = train_positive_indices
            # negative_indices = train_negative_indices
        elif split == "test":
            positive_indices = test_positive_indices
            # negative_indices = test_negative_indices
        else:
            raise ValueError(f"Invalid split: {split}")
        
        np.random.set_state(state)
        
        sample_id = 0 
        for subset_name in tqdm(subsets):
            dataset = datasets.load_dataset("edinburgh-dawg/mmlu-redux", name=subset_name)
            dataset = dataset['test']
            for i in range(len(dataset)):
                if split == "train" and i in train_indices:
                    pass
                elif split == "test" and i in test_indices:
                    pass
                else:
                    continue
                question = dataset[i]['question']
                
                other_answer = [j for j in range(len(dataset[i]['choices'])) if j != dataset[i]['answer']][0]
                other_answer_choice = dataset[i]['choices'][other_answer]
                answer_choice = dataset[i]['choices'][dataset[i]['answer']]
                
                # choices = "Question: {question}\nChoices:\n"
                # for j in range(len(dataset[i]['choices'])):
                #     choices += f"{j}: {dataset[i]['choices'][j]}\n"
                # answer= int(dataset[i]['answer'])
                # assert answer in range(len(dataset[i]['choices']))
                if i in positive_indices:
                    question = question + answer_choice
                    label = 'correct'
                else:
                    question = question + other_answer_choice
                    label = 'incorrect'
                data['question'].append(question)
                data['answer'].append(label)
                data['sample_id'].append(sample_id)
                data['cateogry'].append(subset_name)
                sample_id += 1
        dataset = datasets.Dataset.from_dict(data)
        dataset = dataset.map(lambda x: {"label": convert_label_to_binary(name, x["answer"])})
        return dataset
    
    elif name == "subject_qa":
        CATEGORIES = [
            'books', 'electronics', 'grocery', 'movies', 'restaurants', 'tripadvisor'
        ]

        def read_subjQA_dataset(category, data_cache_dir):
            # Naming rule : subjqa_CATEGORY
            train_csv = pd.read_csv(os.path.join(data_cache_dir,  'SubjQA/SubjQA', category, 'splits/train.csv'))
            dev_csv = pd.read_csv(os.path.join(data_cache_dir,  'SubjQA/SubjQA', category, 'splits/dev.csv'))
            test_csv = pd.read_csv(os.path.join(data_cache_dir, 'SubjQA/SubjQA', category, 'splits/test.csv'))
            return train_csv, dev_csv, test_csv

        data = {
            'train':{
                'question':[],
                'answer':[],
                'category':[],
                'sample_id':[],
            },
            'validation':{
                'question':[],
                'answer':[],
                'category':[],
                'sample_id':[],
            },
            'test':{
                'question':[],
                'answer':[],
                'category':[],
                'sample_id':[],
            },
        }
        NUM_SAMPLES = 500
        for cat in CATEGORIES:
            train_csv, dev_csv, test_csv = read_subjQA_dataset(cat, data_dir)
            for split, csv_data in zip(['train', 'validation', 'test'], [train_csv, dev_csv, test_csv]):
                csv_data['label'] = csv_data['is_ques_subjective'].apply(lambda x: {True:1, False:0}[x])
                data[split]['question'].extend(csv_data.question.to_list()[:NUM_SAMPLES])
                data[split]['answer'].extend(csv_data.label.to_list()[:NUM_SAMPLES])
                data[split]['category'].extend([cat]*NUM_SAMPLES)
                data[split]['sample_id'].extend(list(range(NUM_SAMPLES)))
        dataset = datasets.Dataset.from_dict(data[split])
        dataset = dataset.map(lambda x: {"label": convert_label_to_binary(name, x["answer"])})
        return dataset
        
    else:
        raise ValueError(f"Dataset {name} not found")

def batch_encode_text(list_of_text:List[str], tokenizer, model_name:str=None,):
    """
    Batch encode prompts
    Args:
        tokenizer: tokenizer
        dataset: dataset
    Returns:
        encoded prompts
    """
    # Check if padding_side is already set to "left", if not, set it
    if tokenizer.padding_side != "left":
        print(f"Warning: Tokenizer padding_side was not set to 'left'. Setting it now.")
        tokenizer.padding_side = "left"
        
    batch_encoded = tokenizer(list_of_text, 
                              return_tensors="pt", 
                              padding=True, 
                              truncation=True)
    return batch_encoded

def get_dataloader(dataset, batch_size:int=16):
    """
    Get dataloader
    Args:
        dataset: dataset
        batch_size: batch size
    Returns:
        dataloader
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def convert_label_to_binary(dataset_name:str, label):
    if dataset_name == "sst2-small":
        if label == "positive":
            return 1
        elif label == "negative":
            return 0
        else:
            raise ValueError(f"Label {label} not supported")
    elif dataset_name == "imdb-small":
        if label == "positive":
            return 1
        elif label == "negative":
            return 0
        else:
            raise ValueError(f"Label {label} not supported")
    elif dataset_name == "paradetox":
        if label == "Toxic":
            return 1
        elif label == "Neutral":
            return 0
        else:
            raise ValueError(f"Label {label} not supported")
    elif dataset_name == "truthfulqa":
        if label == "Incorrect":
            return 0
        elif label == "Correct":
            return 1
        else:
            raise ValueError(f"Label {label} not supported")
        
    elif dataset_name in ["deontology", "justice", "virtue"]:
        if label == 0:
            return 0
        elif label == 1:
            return 1
        else:
            raise ValueError(f"Label {label} not supported")
    elif dataset_name == "mmlu":
        if label == "correct":
            return 1
        elif label == "incorrect":
            return 0
        else:
            raise ValueError(f"Label {label} not supported")
    elif dataset_name == "subject_qa":
        if label == 0:
            return 0
        elif label == 1:
            return 1
        else:
            raise ValueError(f"Label {label} not supported")

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
        
        
DATA_LABELS = {
    "sst2-small": [0,1],
    "imdb-small": [0,1],
    "paradetox": [0,1],
    "truthfulqa": [0,1],
    "deontology": [0,1],
    "justice": [0,1],
    "virtue": [0,1],
    'mmlu': [0,1],
    "subject_qa": [0,1],
}