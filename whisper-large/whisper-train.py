import sys
import datetime
import os
import wandb

print("Python version:", sys.version)

base_model = "openai/whisper-large-v3"
base_output_dir = "/opt/training"

hf_token = "xxxx"
dataset_link = "sidleal/TARSILA-ASR-V1"

os.environ["WANDB_API_KEY"] = "xxxx" 
os.environ["WANDB_PROJECT"] = "tarsila-asr-whisper"
os.environ["WANDB_RUN_NAME"] = f"tarsila-asr-whisper-run-1"
wandb.init(project="tarsila-asr-whisper", name=f"tarsila-asr-whisper-run-1")

print("Begin", datetime.datetime.now())

from datasets import load_dataset, DatasetDict, Dataset
tarsila_asr = DatasetDict()

tarsila_asr_temp = load_dataset(dataset_link, split="validation", token=hf_token, streaming=True)
tarsila_asr_subset_temp = [example for i, example in enumerate(tarsila_asr_temp) if i < 1500]

tarsila_asr["train"] = Dataset.from_list(tarsila_asr_subset_temp[:1000])
tarsila_asr["validation"] = Dataset.from_list(tarsila_asr_subset_temp[1000:])

tarsila_asr = tarsila_asr.remove_columns(["origin", "duration", "gender"])

print(tarsila_asr)

print("End Load Dataset", datetime.datetime.now())




from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)

from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained(base_model, language="Portuguese", task="transcribe")

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(base_model, language="Portuguese", task="transcribe")


def prepare_dataset(batch):
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


tarsila_asr = tarsila_asr.map(prepare_dataset, remove_columns=tarsila_asr.column_names["train"], num_proc=2)

print("End Load Dataset", datetime.datetime.now())

from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained(base_model)

model.generation_config.language = "portuguese"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

import evaluate
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


fine_tuned_model_name = "whisper-large-v3-pt-tst"

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=base_output_dir+"/"+fine_tuned_model_name,  # change to a repo name of your choice
    run_name=fine_tuned_model_name,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=500,
    logging_steps=25,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=tarsila_asr["train"],
    eval_dataset=tarsila_asr["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer,
)


processor.save_pretrained(training_args.output_dir)

trainer.train()

kwargs = {
    "language": "pt",
    "model_name": "Whisper Large v3 pt-br test",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-large-v3",
    "tasks": "automatic-speech-recognition",
}

trainer.push_to_hub(**kwargs, token=hf_token)

wandb.finish()

print("End Training", datetime.datetime.now())

