from transformer import pipeline
from faiss import read_index
from retrieval import get_embedding
PIPELINE_NAME = 'question-answering'
MODEL_NAME = 'duynd/distilbert-finetuned-squadv2'
pipe = pipeline(PIPELINE_NAME, model = MODEL_NAME)
vectordb = read_index('index.faiss')


def get_answer(input_question,TOP_K = 5):
    print(f'Input question: {input_question}')
    input_quest_embedding = get_embedding(input_question).cpu().detach().numpy()
    scores, samples = vectordb.search(input_quest_embedding, TOP_K)
    for idx, score in enumerate(scores):
        question = samples["question"][idx]
        context = samples["context"][idx]
        answer = pipe(question,context)
        print(f"Top {idx+1}\t Score: {score}")
        print(f"Context: {context}")
        print(f"Answer: {answer}")

if __name__=="__main__":
    get_answer(input_question=input(""))