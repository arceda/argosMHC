from utils import compute_metrics

# Load model directly
from transformers import AutoTokenizer, EsmModel, EsmConfig, EsmPreTrainedModel
import torch
model_name = "../../pre_trained_models/esm2_t36_3B_UR50D/"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
#model = EsmForMaskedLM.from_pretrained(model_name, local_files_only=True)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True, num_labels=2)

import pandas as pd
cols = ["HLA", "peptide", "label", "length", "mhc"]
df = pd.read_csv("../../dataset/hlab/hlab_test.csv",
                header=1,
                names=cols,
                low_memory=False)

df.drop(["HLA", "length", "mhc"], axis=1, inplace=True)

train_df = df.sample(frac=0.8, random_state=2024)
test_df = df.drop(train_df.index)

max_length = 512
def encode_text(texts):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            #return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

train_input_ids, train_attention_masks = encode_text(train_df["peptide"])
test_input_ids, test_attention_masks = encode_text(test_df["peptide"])

# Obtener las etiquetas de entrenamiento y prueba
train_labels = torch.tensor(train_df["label"].values)
test_labels = torch.tensor(test_df["label"].values)

from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

config = {
        "lora_alpha": 1, 
        "lora_dropout": 0.5,
        "lr": 3.701568055793089e-04,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 36,
        "r": 2,
        "weight_decay": 0.3,
        # Add other hyperparameters as needed
    }

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=config["r"],
    lora_alpha=config["lora_alpha"],
    target_modules=[
            "query",
            "key",
            "value",
            "EsmSelfOutput.dense",
            "EsmIntermediate.dense",
            "EsmOutput.dense",
            "EsmContactPredictionHead.regression",
            "classifier"
    ],
    lora_dropout=config["lora_dropout"],
    bias="none",#all, lora_only
)

model = get_peft_model(model, peft_config)