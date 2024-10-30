from transformers import Trainer, TrainingArguments, BertConfig, AdamW
from model_utils_bert import BertLinear, BertRnn, BertRnnAtt, BertRnnSigmoid, BertRnnDist
from model_utils_tape import TapeLinear, TapeRnn, TapeRnnAtt, TapeRnnDist
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

import wandb
from transformers import set_seed
set_seed(10)
#set_seed(1)

import sys
import argparse


# Ejemplo de uso para empezar un entrenamiento
# python train.py -t bert -c ../checkpoints_train/classic -m ../models/classic -p ../pre_trained_models/esm2_t33_650M_UR50D -r 0 
# python train.py -t bert -c ../checkpoints_train/classic_t33_c3 -m ../models/classic_t33_c3 -p ../pre_trained_models/esm2_t6_8M_UR50D -r 0 
# python train.py -t bert -c ../checkpoints_train/classic_t33_c3 -m ../models/classic_t33_c3 -p ../pre_trained_models/esm2_t33_650M_UR50D -r 0 
# python train.py -t dist -c ../checkpoints_train/esm2_distilbert_t33_c3 -m ../models/esm2_distilbert_t33_c3 -p ../pre_trained_models/esm2_t33_650M_UR50D -r 0
# python train.py -t dist -c ../checkpoints_train/esm2_distilbert_t33_c4 -m ../models/esm2_distilbert_t33_c4 -p ../pre_trained_models/esm2_t33_650M_UR50D -r 0
# python train.py -t dist -c ../checkpoints_train/esm2_distilbert_t33_c5 -m ../models/esm2_distilbert_t33_c5 -p ../pre_trained_models/esm2_t33_650M_UR50D -r 0
# python train.py -t mamba -c ../checkpoints_train/mamba -m ../models/mamba -p ../pre_trained_models/esm2_t33_650M_UR50D -r 0 

# Ejemplo de uso para resumir un entrenamiento
# python train.py -t bert -c ../checkpoints_train/classic -m ../models/classic -p ../pre_trained_models/esm2_t33_650M_UR50D -r 1 -id <wandb_id>

# freeze
#python train.py -t bert -c ../checkpoints_train/esm2_t33_fz_c3 -m ../models/esm2_t33_fz_c3 -p ../pre_trained_models/esm2_t33_650M_UR50D -r 0 
#python train.py -t bert -c ../checkpoints_train/esm2_t33_fz_c3 -m ../models/esm2_t33_fz_c3 -p ../pre_trained_models/esm2_t33_650M_UR50D -r 1 -id po1o9ddn

# prot-bert-bfd
#python train.py -t bert -c ../checkpoints_train/prot_bert_bfd_c5 -m ../models/prot_bert_bfd_c5 -p ../pre_trained_models/prot_bert_bfd -r 0 
#python train.py -t bert -c ../checkpoints_train/prot_bert_bfd_c5 -m ../models/prot_bert_bfd_c5 -p ../pre_trained_models/prot_bert_bfd -r 1 -id bznk10f8

# protbert-bfd-fz
#python train.py -t bert -c ../checkpoints_train/prot_bert_bfd_c5_fz -m ../models/prot_bert_bfd_c5_fz -p ../pre_trained_models/prot_bert_bfd -r 1 -id 0uua2u5k


parser = argparse.ArgumentParser(prog='pMHC')
parser.add_argument('-t', '--type', default='bert', help='Model type: tape or bert')     
parser.add_argument('-c', '--checkpoints', default='results/tmp/', help='Path to store results')  
parser.add_argument('-m', '--models', default='models/tmp/', help='Path to store models')  
parser.add_argument('-p', '--pretrained', default='pre_trained_models/esm2_t6_8M_UR50D', help='Pretrained model path')  
parser.add_argument('-r', '--resume', help='Resume training')
parser.add_argument('-id', '--identification', default=None, help='Identification of runtime wandb')

args = parser.parse_args()
model_type          = args.type         # tape or esm2( for esm2 and protbert)
path_checkpoints    = args.checkpoints  # path to store checkpoints
path_model          = args.models       # path to save the best model
model_name          = args.pretrained   # path of the pre-trained model, for esm2 and protbert
resume              = int(args.resume)       # boolean, if true the training will resume from last checkpoint
wandb_ide           = args.identification   # wandb id, it is use when the training is resumed

print("Training :", model_type, path_checkpoints, path_model, model_name, resume, wandb_ide)

# wandb  ###########################################################################################
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
if not resume:
    run = wandb.init(project="argosMHC")
else:
    run = wandb.init(project="argosMHC", id=wandb_ide, resume="must") # el id esta en la wandb


# dataset ###########################################################################3###############
path_train_csv = "../datasets/hlab/hlab_train.csv"
path_val_csv = "../datasets/hlab/hlab_val.csv"
max_length = 50 # for hlab dataset
#max_length = 73 # for netpanmhcii3.2 dataset La longitus del mhc es 34 => 34 + 37 + 2= 73  

# model   ###########################################################################3###############

if model_type == "tape":  
    trainset = DataSetLoaderTAPE(path_train_csv, max_length=max_length) 
    valset = DataSetLoaderTAPE(path_val_csv, max_length=max_length)
    config = ProteinBertConfig.from_pretrained(model_name, num_labels=2)

elif model_type == "dist":
    trainset = DataSetLoaderBERT(path=path_train_csv, tokenizer_name=model_name, max_length=max_length)
    valset = DataSetLoaderBERT(path=path_val_csv, tokenizer_name=model_name, max_length=max_length)    
    config = BertConfig.from_pretrained(model_name, num_labels=2)
    
elif model_type == "bert": 
    trainset = DataSetLoaderBERT(path=path_train_csv, tokenizer_name=model_name, max_length=max_length)
    valset = DataSetLoaderBERT(path=path_val_csv, tokenizer_name=model_name, max_length=max_length)    
    config = BertConfig.from_pretrained(model_name, num_labels=2)

elif model_type == "mamba": 
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

if model_type == "tape":    
    model_ = TapeRnn.from_pretrained(model_name, config=config)
elif model_type == "dist":
    model_ = BertRnnDist.from_pretrained(model_name, config=config)
elif model_type == "bert":                       
    model_ = BertRnn.from_pretrained(model_name, config=config)
elif model_type == "mamba":                       
    model_ = BertRnnMamba.from_pretrained(model_name, config=config)


# FREEZE BERT LAYERS ############################################################
for param in model_.bert.parameters():
    param.requires_grad = False
    
############ hyperparameters ESM2 (fails) ####################################### c5
num_samples = len(trainset)
num_epochs = 60 # ***
batch_size = 16  

weight_decay = 0.01
lr =2e-6 # ***
betas = ((0.9, 0.98)) 
num_training_steps = int((num_epochs * num_samples)/batch_size) 
# warmup_steps = int(num_training_steps*0.1) # before
warmup_steps = 202132 # now


training_args = TrainingArguments(
        output_dir                  = path_checkpoints, 
        num_train_epochs            = num_epochs,   
        per_device_train_batch_size = batch_size,   
        per_device_eval_batch_size  = batch_size * 32,         
        logging_dir                 = path_checkpoints,        
        logging_strategy            = "steps", #epoch or steps
        #eval_steps                  = num_samples/batch_size, # para epochs
        #save_steps                  = num_samples/batch_size, # para epochs
        eval_steps                  = 20000, # el primer experimento fue con 1000 steps
        save_steps                  = 20000,
        metric_for_best_model       = 'f1',
        load_best_model_at_end      = True,        
        evaluation_strategy         = "steps", #epoch or steps
        save_strategy               = "steps", #epoch or ste
        #gradient_accumulation_steps = 64,  # reduce el consumo de memoria
    
        report_to="wandb",
        logging_steps=20000  # how often to log to W&B
)



optimizer = AdamW(model_.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, correct_bias=True)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

trainer = Trainer(        
        args            = training_args,   
        model           = model_, 
        train_dataset   = trainset,  
        eval_dataset    = valset, 
        compute_metrics = compute_metrics,  
        optimizers      = (optimizer, lr_scheduler),      
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)] 
    )


if not resume:
    trainer.train()
else:
    trainer.train(resume_from_checkpoint = True)

trainer.save_model(path_model)
wandb.finish()



