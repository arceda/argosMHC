from transformers import Trainer, TrainingArguments, BertConfig, AdamW
from model_utils_bert import BertLinear, BertRnn, BertRnnAtt, BertRnnSigmoid
#from model_utils_tape import TapeLinear, TapeRnn, TapeRnnAtt, TapeRnnDist
from utils import compute_metrics
from transformers import EarlyStoppingCallback, IntervalStrategy

from tape import ProteinBertConfig
from torch.utils.data import DataLoader
from transformers import get_scheduler, TrainerCallback

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


# data loaders
from dataloader_bert import DataSetLoaderBERT, DataSetLoaderBERT_old

#from dataloader_tape import DataSetLoaderTAPE

import wandb
from transformers import set_seed
set_seed(10)
#set_seed(1)

import sys
import argparse


# Ejemplo de uso para empezar un entrenamiento
# C5
# python train_lora.py -t bert -c /notebooks/checkpoints_train/lora_t33_c5_jun1 -m /notebooks/models/lora_t33_c5_jun1 -p /notebooks/pre_trained_models/esm2_t33_650M_UR50D -r 0 

# Ejemplo de uso para resumir un entrenamiento
# C5
# python train_lora.py -t bert -c /notebooks/checkpoints_train/lora_t33_c5_jun1 -m /notebooks/models/lora_t33_c5_jun1 -p /notebooks/pre_trained_models/esm2_t33_650M_UR50D -r 1 -id 
#python train_lora.py -t bert -c ../checkpoints_train/lora_t33_c3.1 -m ../models/lora_t33_c3.1 -p ../pre_trained_models/esm2_t33_650M_UR50D -r 1 -id gdcdmn3d
#python train_lora.py -t bert -c ../checkpoints_train/lora_t33_c4 -m ../models/lora_t33_c4 -p ../pre_trained_models/esm2_t33_650M_UR50D -r 1 -id y725v892
#python train_lora.py -t bert -c ../checkpoints_train/lora_t33_c5 -m ../models/lora_t33_c5 -p ../pre_trained_models/esm2_t33_650M_UR50D -r 1 -id jzxt25ku


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
path_train_csv = "/notebooks/datasets/hlab/hlab_train.csv"
path_val_csv = "/notebooks/datasets/hlab/hlab_val.csv"
max_length = 50 # for hlab dataset
#max_length = 73 # for netpanmhcii3.2 dataset La longitus del mhc es 34 => 34 + 37 + 2= 73  

# model   ###########################################################################3###############

if model_type == "tape":  
    trainset = DataSetLoaderTAPE(path_train_csv, max_length=max_length) 
    valset = DataSetLoaderTAPE(path_val_csv, max_length=max_length)
    config = ProteinBertConfig.from_pretrained(model_name, num_labels=2)

if model_type == "dist":
    trainset = DataSetLoaderTAPE(path_train_csv, max_length=max_length) 
    valset = DataSetLoaderTAPE(path_val_csv, max_length=max_length)
    config = ProteinBertConfig.from_pretrained(model_name, num_labels=2)
    
if model_type == "bert": 
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
    model_ = TapeRnnDist.from_pretrained(model_name, config=config)
if model_type == "bert":                       
    model_ = BertRnn.from_pretrained(model_name, config=config)


############ hyperparameters #################################################### Configuration 5
num_samples = len(trainset)
num_epochs = 60 # ***
batch_size = 16  

weight_decay = 0.01
lr =2e-6 # ***
betas = ((0.9, 0.98)) 
num_training_steps = int((num_epochs * num_samples)/batch_size) 
# warmup_steps = int(num_training_steps*0.1) # before
warmup_steps = 202132 # now

# LoRA config ####################################################################
configLora = { "lora_alpha": 1, "lora_dropout": 0.4, "r": 1 }
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, 
    inference_mode=False, 
    r=configLora["r"], 
    lora_alpha=configLora["lora_alpha"], 
    target_modules=["query", "key", "value"], # also maybe "dense_h_to_4h" and "dense_4h_to_h"
    lora_dropout=configLora["lora_dropout"], 
    bias="none" # or "all" or "lora_only" 
)

model_ = get_peft_model(model_, peft_config)
#model_ = accelerator.prepare(model_)
model_.print_trainable_parameters()

# trainable params: 256,514 || all params: 679,378,692 || trainable%: 0.0378

# FREEZE BERT LAYERS ############################################################
#for param in model_.bert.parameters():
#    param.requires_grad = False
    

training_args = TrainingArguments(
        output_dir                  = path_checkpoints, 
        num_train_epochs            = num_epochs,   
        per_device_train_batch_size = batch_size,   
        per_device_eval_batch_size  = batch_size * 32,   # before       batch_size * 8 
        logging_dir                 = path_checkpoints,        
        logging_strategy            = "steps", #epoch or steps
        #eval_steps                  = num_samples/batch_size, # para epochs
        #save_steps                  = num_samples/batch_size, # para epochs
        eval_steps                  = 10000, # el primer experimento fue con 1000 steps
        save_steps                  = 10000,
        metric_for_best_model       = 'f1',
        load_best_model_at_end      = True,        
        evaluation_strategy         = "steps", #epoch or steps
        save_strategy               = "steps", #epoch or ste
        #gradient_accumulation_steps = 64,  # reduce el consumo de memoria
    
        report_to="wandb",
        logging_steps=10000  # how often to log to W&B
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
trainer.model.config.save_pretrained(path_model)
wandb.finish()



