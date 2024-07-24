from huggingface_hub import login
import os
import numpy as np
from tqdm.auto import tqdm
import collections
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from transformers import Trainer
import evaluate
from utils.reader import preprocess_training_examples, preprocess_validation_examples
from .evaluate_reader import compute_metrics
device = torch.device ("cuda") if torch.cuda.is_available() else torch.device("cpu")

huggingface_token = os.environ['huggingface_token']
login(huggingface_token)

MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 384
STRIDE = 128
DATASET_NAME = 'squad_v2'
raw_datasets = load_dataset(DATASET_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = raw_datasets["train"].map(preprocess_training_examples, batched = True, remove_columns = raw_datasets["train"].column_names)
validation_dataset = raw_datasets["validation"].map(preprocess_validation_examples, batched = True, remove_columns = raw_datasets["validation"].column_names)


model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
args = TrainingArguments(
    output_dir = "distilbert-finetuned-squadv2",
    evaluation_strategy = "no",
    save_strategy = "epoch",
    learning_rate = 2e-5,
    num_train_epochs = 3,
    weight_decay = 0.01,
    fp16=True,
    push_to_hub = True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset = train_dataset,
    eval_dataset = validation_dataset,
    tokenizer = tokenizer,
)
trainer.train()

#Evaluate the model
predictions, _, _ = trainer.predict(validation_dataset)
start_logits = predictions.start_logits
end_logits = predictions.end_logits
compute_metrics(predictions.start_logits, predictions.end_logits, validation_dataset, raw_datasets["validation"])

trainer.push_to_hub(commit_message = "Training complete")