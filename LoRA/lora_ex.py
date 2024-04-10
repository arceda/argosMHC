#Comandos previos
# pip install -q transformers
# pip install -q peft
# pip install -q evaluate

from datasets import load_dataset
dataset = load_dataset("imdb")

###############################################################################
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

##############################################################################
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

##############################################################################
# PEFT: Parametros de ajuste fino eficiente
# task_type: Especifica el tipo de tarea el cual  el modelo se ajustara
# r son las dimensiones de las matrices A y B
# lora_alpha es el factor de escala, determina la relacion de los pesos A y B
# en relacion con los parametros originales del modelo
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1
)

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-cased',
    num_labels=2
)

##############################################################################
# Insertamos las matrices A y B en el modelo (get_peft_model)
from peft import get_peft_model
model = get_peft_model(model, lora_config)

##############################################################################
# ENTRENAMIENTO Y EVALUACION DEL MODELO
# evaluate.load calcula e informa metricas, es una fx de precision sencilla
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

##############################################################################
# Calculamos la precision de las predicciones
# Antes de pasar las predicciones para calcular, convertimos los logits
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

##############################################################################
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",
                                 num_train_epochs=13,)

##############################################################################
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

##############################################################################
trainer.train()

trainer.save_model("./BERT_LoRA")
#tokenizer.save_pretrained("BERT_LoRA_tokenizer")