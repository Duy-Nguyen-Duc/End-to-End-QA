
from transformers import AutoTokenizer
MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 384
STRIDE = 128
DATASET_NAME = 'squad_v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# preprocess training data
def preprocess_training_examples(examples):
  questions = [q.strip() for q in examples["question"]]
  inputs = tokenizer(
      questions,
      examples["context"],
      max_length=MAX_LENGTH,
      truncation="only_second",
      stride=STRIDE,
      return_overflowing_tokens=True,
      return_offsets_mapping=True,
      padding="max_length",
  )
  offset_mapping = inputs.pop("offset_mapping")
  sample_map = inputs.pop("overflow_to_sample_mapping")
  answers = examples["answers"]
  start_positions = []
  end_positions = []
  for i, offset in enumerate(offset_mapping):
    sample_idx = sample_map[i]
    sequence_ids = inputs.sequence_ids(i)
    idx = 0
    while sequence_ids[idx] != 1:
      idx += 1
    context_start= idx
    while sequence_ids[idx] == 1:
      idx += 1
    context_end = idx - 1

    answer = answers[sample_idx]
    if len(answer["answer_start"]) == 0:
      start_positions.append(0)
      end_positions.append(0)
    else:
      start_char = answer["answer_start"][0]
      end_char = answer["answer_start"][0] + len(answer["text"][0])
      if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
      else:
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
          idx += 1
        start_positions.append(idx - 1)
        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
          idx += 1
        end_positions.append(idx + 1)
  inputs["start_positions"] = start_positions
  inputs["end_positions"] = end_positions
  return inputs


def preprocess_validation_examples(examples):
  questions = [q.strip() for q in examples["question"]]
  inputs = tokenizer(
      questions,
      examples["context"],
      max_length=MAX_LENGTH,
      truncation="only_second",
      stride=STRIDE,
      return_overflowing_tokens=True,
      return_offsets_mapping=True,
      padding="max_length",
  )
  sample_map = inputs.pop("overflow_to_sample_mapping")
  example_ids = []
  for i in range(len(inputs["input_ids"])):
    sample_idx = sample_map[i]
    example_ids.append(examples["id"][sample_idx])

    sequence_ids = inputs.sequence_ids(i)
    offset = inputs["offset_mapping"][i]
    inputs["offset_mapping"][i] = [
        o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
    ]
  inputs["example_id"] = example_ids
  return inputs