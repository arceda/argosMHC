# Predictions for HLAB dataset for TAPE

# load model
from model_utils_bert import BertRnn, BertRnnDist
from transformers import Trainer, TrainingArguments, BertConfig
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from dataloader_bert import DataSetLoaderBERT
from transformers import BertConfig
from utils import compute_metrics
import json

#model_name = "results/train_tape_rnn_acc_steps/checkpoint-1578"
#model_name = "results/train_tape_rnn_freeze_acc_steps_30_epochs/checkpoint-9468"
#model_name = "results/train_tape_rnn_acc_steps/checkpoint-1578"
model_name = "/M2/ArgosMHC_backup/classic_t33_c5/classic_t33_c5/checkpoint-153000/"  # mejor checkpoiunt
name_results = "predictions_esm2_classic_t33_c5" # nombre de los archivos donde se guardara los resultados. 

seq_length = 50 # for MHC-I
config = BertConfig.from_pretrained(model_name, num_labels=2 )
print(config)

model = Trainer(model = BertRnn.from_pretrained(model_name, config=config), compute_metrics = compute_metrics)
test_dataset = DataSetLoaderBERT("/home/vicente/projects/pmhc/dataset/hlab/hlab_test.csv", tokenizer_name="/home/vicente/projects/pmhc/pre_trained_models/esm2_t33_650M_UR50D", max_length=seq_length)
predictions, label_ids, metrics = model.predict(test_dataset)
print(model_name)
print(metrics)
f = open(name_results + ".txt", "w")
f.write(model_name + "\n")
f.write(json.dumps(metrics))
f.close()

########################## print predictions #######################################
####################################################################################
import pandas as pd
df = pd.DataFrame(predictions)

df['prediction'] = df.apply(lambda row: ( 0 if row[0] > row[1] else 1 ), axis=1)
df['label'] = label_ids
print(df)
df.to_csv(name_results + ".csv")