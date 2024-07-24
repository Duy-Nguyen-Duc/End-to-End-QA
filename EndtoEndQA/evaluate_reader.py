import evaluate
import collections
from tqdm import tqdm
import numpy as np

metric = evaluate.load("squad_v2")
N_BEST = 20
MAX_ANSWER_LENGTH = 30
def compute_metrics(start_logits, end_logits, features, examples):
  example_to_features = collections.defaultdict(list)
  for idx, feature in enumerate(features):
    example_to_features[feature["example_id"]].append(idx)
  predicted_answers = []
  for example in tqdm(examples):
    example_id = example["id"]
    context = example["context"]
    answers = []
    for feature_index in example_to_features[example_id]:
      start_logit = start_logits[feature_index]
      end_logit = end_logits[feature_index]
      offsets = features[feature_index]["offset_mapping"]
      start_indexes = np.argsort(start_logit)[-1:N_BEST-1:-1].tolist()
      end_indexes = np.argsort(end_logit)[-1:N_BEST-1:-1].tolist()
      for start_index in start_indexes:
        for end_index in end_indexes:
          if offsets[start_index] is None or offsets[end_index] is None:
            continue
          if end_index - start_index +1 > MAX_ANSWER_LENGTH:
            continue
          text = context[
              offsets[start_index][0]: offsets[end_index][1]
          ]
          logit_score = start_logit[start_index] + end_logit[end_index]
          answer = {
              "text": text,
              "logit_score": logit_score
          }
          answers.append(answer)
    if len(answers) > 0:
      best_answer = max(answers, key=lambda x: x["logit_score"])
      answer_dict = {
          'id': example_id,
          'prediction_text': best_answer['text'],
          'no_answer_probability': 1-best_answer['logit_score']
      }
    else:
      answer_dict = {
          'id': example_id,
          'prediction_text': '',
          'no_answer_probability': 1.0
      }
    predicted_answers.append(answer_dict)
  theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
  return metric.compute(predictions=predicted_answers, references=theoretical_answers)

