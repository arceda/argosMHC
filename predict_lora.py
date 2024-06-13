
# Predictions for HLAB dataset for TAPE

# load model
from model_utils_bert import BertRnn, BertRnnDist
from transformers import Trainer, TrainingArguments, BertConfig
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from dataloader_bert import DataSetLoaderBERT
from transformers import BertConfig, AutoModel
from utils import compute_metrics
from accelerate import Accelerator
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import json

#model_name = "results/train_tape_rnn_acc_steps/checkpoint-1578"
#model_name = "results/train_tape_rnn_freeze_acc_steps_30_epochs/checkpoint-9468"
#model_name = "results/train_tape_rnn_acc_steps/checkpoint-1578"
model_name = "../checkpoints_train/lora_t33_c3_2/checkpoint-150000/"  # mejor checkpoiunt
name_results = "predictions_esm2_lora_t33_c3" # nombre de los archivos donde se guardara los resultados. 


#model = AutoModel.from_pretrained(model_name, revision="v2.0.1")
#print(model)

seq_length = 50 # for MHC-I
config = BertConfig.from_pretrained(model_name, num_labels=2 )
print(config)

'''peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, 
    inference_mode=False, 
    r=1, 
    lora_alpha=1, 
    target_modules=["query", "key", "value"], # also maybe "dense_h_to_4h" and "dense_4h_to_h"
    lora_dropout=0.4, 
    bias="none" # or "all" or "lora_only" 
)

model_ = get_peft_model(model_, peft_config)
model_ = accelerator.prepare(model_)
'''

model = Trainer(model = BertRnn.from_pretrained(model_name, config=config), compute_metrics = compute_metrics)
test_dataset = DataSetLoaderBERT("../datasets/hlab/hlab_test.csv", tokenizer_name="../pre_trained_models/esm2_t33_650M_UR50D", max_length=seq_length)
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

