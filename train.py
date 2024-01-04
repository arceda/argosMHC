from transformers import Trainer, TrainingArguments, BertConfig, AdamW
from model_utils_bert import BertLinear, BertRnn, BertRnnAtt, BertRnnSigmoid
from model_utils_tape import TapeLinear, TapeRnn, TapeRnnAtt
from utils import compute_metrics
from transformers import EarlyStoppingCallback, IntervalStrategy

from tape import ProteinBertConfig
from torch.utils.data import DataLoader
from transformers import get_scheduler, TrainerCallback

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

# data loaders
from dataloader_bert import DataSetLoaderBERT, DataSetLoaderBERT_old
from dataloader_tape import DataSetLoaderTAPE

from transformers import set_seed
set_seed(42)
#set_seed(1)

import sys
import argparse

parser = argparse.ArgumentParser(prog='pMHC')
parser.add_argument('-t', '--type', default='bert', help='Model type: tape or bert')     
parser.add_argument('-r', '--results', default='results/tmp/', help='Path to store results')  
parser.add_argument('-m', '--models', default='models/tmp/', help='Path to store models')  
parser.add_argument('-p', '--pretrained', default='pre_trained_models/esm2_t6_8M_UR50D', help='Pretrained model path')  

args = parser.parse_args()
model_type      = args.type
path_results    = args.results
path_model      = args.models
model_name      = args.pretrained

print("Training :", model_type, path_results, path_model, model_name)

# dataset
path_train_csv = "dataset/hlab/hlab_train.csv"
path_val_csv = "dataset/hlab/hlab_val.csv"

#################################################################################
#################################################################################
# MODEL TYPE: bert or tape
#model_type = "tape"
#model_type = "bert" # EM1, ESM2, PortBert

# PATH FOR RESULTS AND MODELS
#path_results    = "results/train_protbert_bfd_rnn/" 
#path_model      = "models/train_protbert_bfd_rnn/"
#path_results    = "results/train_esm2_t6_rnn_30_epochs/" 
#path_model      = "models/train_esm2_t6_rnn_30_epochs/"
#path_results    = "results/train_esm2_t30_rnn10/" # plotgradients, evaluated each 1000 steps
#path_model      = "models/train_esm2_t30_rnn10/"
#path_results    = "results/train_esm2_t30_rnn11/" # plotgradients, evaluated each 100 optimization steps, plot gradients each 64 spteps. Se agrego gradient accumulation steps 64
#path_model      = "models/train_esm2_t30_rnn11/"
#path_results    = "results/train_esm2_t30_rnn12/" # plotgradients, evaluated each 100 optimization steps, plot gradients each 100 optimization spteps. Se agrego gradient accumulation steps 64
#path_model      = "models/train_esm2_t30_rnn12/"
#path_results    = "results/train_esm2_t33_rnn_freeze_acc_steps/" 
#path_model      = "models/train_esm2_t33_rnn_freeze_acc_steps/"
#path_results    = "results/train_esm2_t6_rnn2/" # con trainner plot gradients, todo bien :) 
#path_model      = "models/train_esm2_t6_rnn2/"


# PRETRAINED MODELS
#model_name = "bert-base"   # TAPE                   
#model_name = "pre_trained_models/esm2_t6_8M_UR50D"         
#model_name = "pre_trained_models/esm2_t12_35M_UR50D" 
#model_name = "pre_trained_models/esm2_t30_150M_UR50D"
#model_name = "pre_trained_models/esm2_t33_650M_UR50D"       
#model_name = "pre_trained_models/prot_bert_bfd"
#################################################################################
#################################################################################

max_length = 50 # for hlab dataset
#max_length = 73 # for netpanmhcii3.2 dataset La longitus del mhc es 34 => 34 + 37 + 2= 73  

if model_type == "tape":  
    trainset = DataSetLoaderTAPE(path_train_csv, max_length=max_length) 
    valset = DataSetLoaderTAPE(path_val_csv, max_length=max_length)
    config = ProteinBertConfig.from_pretrained(model_name, num_labels=2)
    
else: 
    trainset = DataSetLoaderBERT(path=path_train_csv, tokenizer_name=model_name, max_length=max_length)
    valset = DataSetLoaderBERT(path=path_val_csv, tokenizer_name=model_name, max_length=max_length)    
    config = BertConfig.from_pretrained(model_name, num_labels=2)



config.rnn = "lstm"
config.num_rnn_layer = 2
config.rnn_dropout = 0.1
config.rnn_hidden = 768
config.length = max_length
config.cnn_filters = 512
config.cnn_dropout = 0.1

#################################################################################
#################################################################################
if model_type == "tape":    
    model_ = TapeRnn.from_pretrained(model_name, config=config)
else:                       
    model_ = BertRnn.from_pretrained(model_name, config=config)


# FREEZE BERT LAYERS
#for param in model_.bert.parameters():
#    param.requires_grad = False
    
#################################################################################
#################################################################################

#dataset = DataLoader(trainset)
#iterator = iter(dataset)
#print(next(iterator))
#print(next(iterator))
#print(trainset[0]['input_ids'].shape)

############ hyperparameters ####################################################

num_samples = len(trainset)
num_epochs = 3
batch_size = 16  # segun hlab, se obtienen mejoes resutlados
num_training_steps = num_epochs * num_samples


training_args = TrainingArguments(
        output_dir                  = path_results, 
        num_train_epochs            = num_epochs,   
        per_device_train_batch_size = batch_size,   
        per_device_eval_batch_size  = batch_size * 8,         
        logging_dir                 = path_results,        
        logging_strategy            = "epoch", #epoch or steps
        eval_steps                  = num_samples/batch_size, 
        save_steps                  = num_samples/batch_size,  
        #eval_steps                  = 100, # cada 500 optimization steps !!! esto es distinto a los steps
        #save_steps                  = 100, # esto solo es  necesario cuando evaluamos cada 1000 steps
        metric_for_best_model       = 'f1',
        load_best_model_at_end      = True,        
        evaluation_strategy         = "epoch", #epoch or steps
        save_strategy               = "epoch", #epoch or steps
        #debug="debug underflow_overflow"
    
        # cambios para tratar de evitar vanish gradients
        #gradient_accumulation_steps = 64,  # total number of steps before back propagation
        #gradient_accumulation_steps = 128,  # total number of steps before back propagation
        #gradient_accumulation_steps = 200,  # total number of steps before back propagation
        #fp16                        = True,  # Use mixed precision
        #fp16_opt_level              = "02",  # mixed precision mode

    )


# hiperparameters BERTMHC, uso SGD with momentum, ademas uso escheduler
#lr = 0.15  # con este learning rate, no converge y genera Nan
weight_decay = 0.0001
momentum = 0.9
#warmup_steps = 0

# hiperparameters HLAB, uso AdamW (en teoria es mejor de SGD with momentum)
lr = 5e-5  #-> este se uso en todos los experimentos
#lr = 2e-5  -> train_esm2_t30_rnn5 NO converge
#weight_decay = 0.01
betas = ((0.9, 0.999)) # defult
warmup_steps = 1000


#################################### parameters of ESM2 #################################
# segun el paper, los modelos grandes en 270K steps, the bigger model es mejor que los modelos pequeños
# No converge
"""
- ADAM b1 = 0.9, b2 = 0.98
- e = 10-8
- weight decay = 0.01; 0.1 for model of 15 billion parameters
- warm up e = 2000 steps to 4e-4 (1.6e-4 for 15B parameters)
- scheduller linearly decay 
"""
#lr = 4e-4
#weight_decay = 0.01
#betas = ((0.9, 0.98)) # defult
#warmup_steps = 2000


# optimizer Adam Weigh Decay https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
optimizer = AdamW(model_.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

# scheduller
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

trainer = Trainer(        
        args            = training_args,   
        model           = model_, 
        train_dataset   = trainset,  
        eval_dataset    = valset, 
        compute_metrics = compute_metrics,  
        optimizers      = (optimizer, lr_scheduler),        
        #callbacks       = [EarlyStoppingCallback(early_stopping_patience=5), MyCallback()] 
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)] 
    )

#trainer.train(resume_from_checkpoint = True)
trainer.train()
trainer.save_model(path_model)


# python train.py -t tape -r results -m models -p bert-base

