import torch
from datasets import load_dataset
from transformer import AutoTokenizer, AutoModel
from faiss import add_faiss_index, write_index
device = torch.device ("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATASET_NAME = 'squad_v2'
raw_datasets = load_dataset(DATASET_NAME)
raw_datasets = raw_datasets.filter(lambda x: len(x["answers"]["text"]) > 0)
MODEL_NAME = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embedding(text_list):
    encoded_input = tokenizer(
        text_list,
        padding = True,
        truncation = True,
        return_tensors = 'pt'
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

EMBEDDING_COLUMN = 'question_embedding'
embeddings_dataset = raw_datasets.map(
    lambda x: {
        EMBEDDING_COLUMN: get_embedding(x['question']).detach().cpu().numpy()[0]
        }
    )

embeddings_dataset.add_faiss_index(column = EMBEDDING_COLUMN)
faiss_index = embeddings_dataset.get_index(EMBEDDING_COLUMN)

faiss.write_index(faiss_index, 'index.faiss')
