import os
import pandas as pd
import pickle
import re
import string
import subprocess
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

import sys
sys.path.insert(0, 'multienc_bart_scripts_v2/')
sys.modules['transformers.generation_utils'] = __import__('generation_utils_custom')

from modeling_custom_bart import BartForConditionalGeneration
from transformers import BartTokenizer

# set by hands
source_lens = {
    'history': 600,
    'history_aug': 600,
    'history_aug_disco': 600,
    'history_amr': 1024,
    'history_discourse': 40,
    'addr_amr': 300,
    'response': 160,
    'response_aug': 160,
    'response_disco': 160,
    'grounding': 850,
    'title': 64,
    "history_aug#title#grounding": 1024,
    "history_aug_disco#title#grounding": 1024,
    "history#title#grounding": 1024,
    "history#title#grounding": 1024,
    "history#title": 600
}


def generate_top(input_texts, num_beams=4,  max_source_lens=[300, 200, 100], max_target_length=64,
                 top_k=50, top_p=1, temperature=1., do_sample=False):
    model_inputs_list = [tokenizer([input_texts[j]], max_length=max_source_lens[j],
                          padding="max_length" , truncation=True) for j in range(len(input_texts))]

    model_inputs = {}
    for key in model_inputs_list[0]:
        model_inputs[key] = [sum(el, []) for el in zip(*[model_inputs_list[j][key] for j in range(len(model_inputs_list))])]

        
        
    input_ids = torch.LongTensor(model_inputs['input_ids']).to(device)
    
    summary_ids = model.generate(input_ids, do_sample=do_sample,num_beams=num_beams,
                                 max_length=max_target_length, top_k=top_k, top_p=top_p, temperature=temperature)
    pred = tokenizer.batch_decode(summary_ids, clean_up_tokenization_spaces=False)[0]
    pred = re.sub(r'\s+', ' ', pred).replace('</s>', '').replace('<s>', '').strip()
    return pred


def run_train(source_cols=["history_aug#title#grounding", "title", "grounding"],
              target_col="response_aug",
              max_encoder_length=256,
              n_epochs=10, learning_rate=3e-5, batch_size=8, gradient_accumulation_steps=1, device='cuda:0'):
    max_source_lengths = "@".join([str(source_lens[c]) for c in source_cols])
    max_target_length = source_lens[target_col]
    text_columns = "@".join(source_cols)
    
    checkpoint_path = f"checkpoint/multiencoder_bart_v2_bs{batch_size*gradient_accumulation_steps}_{n_epochs}ep_lr{learning_rate}_enclen{max_encoder_length}__from:{'-'.join(source_cols)}___to:{target_col}"
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device.replace('cuda:', '')
    
    subprocess.call(
        ["python", "multienc_bart_scripts_v2/run_summarization.py",
        "--model_name_or_path", "facebook/bart-base",
        "--train_file", "bart_input/train_reddit_dial_df_multi_extented_filt.csv",
        "--validation_file", "bart_input/val_reddit_dial_df_multi_extented_filt.csv",
        "--text_columns", text_columns,
        "--summary_column", target_col,
        "--pad_to_max_length",
        "--max_source_lengths", max_source_lengths,
        "--max_encoder_length", str(max_encoder_length),
        "--max_target_length", str(max_target_length),
        "--do_train",
        "--do_eval", 
        "--per_device_train_batch_size", str(batch_size),
        "--per_device_eval_batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(learning_rate),
        "--save_steps", "20000",
        "--num_train_epochs", str(n_epochs),
        "--output_dir", checkpoint_path,
        "--overwrite_output_dir",
        "--device", device]
    )
    
    return checkpoint_path

def run_eval(model_name_or_path, device='cuda:0'):
    tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(model_name_or_path).train(False)
    
    model.to(device)
    
    source_cols = model_name_or_path.split('/')[1].split('__')[1][5:].split('-')
    target_col = model_name_or_path.split('/')[1].split('__')[-1][4:]
    max_encoder_length = int(re.findall(r'enclen(\d+)\D', model_name_or_path)[0])

    max_source_lengths = [source_lens[c] for c in source_cols]
    max_target_length = source_lens[target_col]
    
    test_data = pd.read_csv("bart_input/val_reddit_dial_df_multi_extented_upd_v2.csv", sep='\t')
    preds = []

    for idx in tqdm(range(len(test_data))):
        input_texts = list(test_data[source_cols].values[idx])
        try:
            pred = generate_top(input_texts,
                                num_beams=1,
                                max_source_lens=max_source_lengths,
                                max_target_length=max_target_length)
        except:
            pred = ""

        preds.append(pred)
        
    with open(model_name_or_path.replace('checkpoint/', 'predictions/').replace('/checkpoint', '_') + '.pkl', 'wb') as f:
        pickle.dump(preds, f)
        