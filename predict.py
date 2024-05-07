# Predictions for HLAB dataset for TAPE

# load model
from model_utils_bert import BertRnn
from transformers import Trainer, TrainingArguments, BertConfig
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from dataloader_tape import DataSetLoaderTAPE
from tape import ProteinBertConfig
from utils import compute_metrics

#model_name = "results/train_tape_rnn_acc_steps/checkpoint-1578"
#model_name = "results/train_tape_rnn_freeze_acc_steps_30_epochs/checkpoint-9468"
#model_name = "results/train_tape_rnn_acc_steps/checkpoint-1578"
model_name = "../pre_trained_models/esm2_t36_3B_UR50D/"
seq_length = 50 # for MHC-I
config = ProteinBertConfig.from_pretrained(model_name, num_labels=2 )
print(config)

model = Trainer(model = BertRnn.from_pretrained(model_name, config=config), compute_metrics = compute_metrics)
test_dataset = DataSetLoaderBERT("dataset/hlab/hlab_test_micro.csv", max_length=seq_length)

predictions, label_ids, metrics = model.predict(test_dataset)
print(model_name)
print(metrics)

########################## print predictions #######################################
####################################################################################
import pandas as pd
df = pd.DataFrame(predictions)

df['prediction'] = df.apply(lambda row: ( 0 if row[0] > row[1] else 1 ), axis=1)
df['label'] = label_ids
print(df)
df.to_csv("tape_rnn_acc_steps.csv")