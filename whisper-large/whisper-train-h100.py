import sys
import datetime
import os
import wandb

print("Python version:", sys.version)

base_model = "openai/whisper-large-v3"
base_output_dir = "/home/jovyan/training"

fine_tuned_model_name = "whisper-large-v3-pt-tst"

hf_token = "xxxx"
dataset_link = "sidleal/TARSILA-ASR-V1"

os.environ["WANDB_API_KEY"] = "f7421cf9b7e382afe518530290443943f1a2543a" 
os.environ["WANDB_PROJECT"] = "tarsila-asr-whisper"
os.environ["WANDB_RUN_NAME"] = f"tarsila-asr-whisper-run-3"
wandb.init(project="tarsila-asr-whisper", name=f"tarsila-asr-whisper-run-3")

print("Begin", datetime.datetime.now())

import itertools
from datasets import load_dataset, DatasetDict, Dataset, Audio
tarsila_asr = DatasetDict()

tarsila_asr["train"] = load_dataset(dataset_link, split="train", token=hf_token)
tarsila_asr["validation"] = load_dataset(dataset_link, split="validation", token=hf_token)

#tarsila_asr_temp = load_dataset(dataset_link, split="validation", token=hf_token, streaming=True)
#tarsila_asr["train"] = tarsila_asr_temp.take(1000)
#tarsila_asr["validation"] = tarsila_asr_temp.skip(1000).take(500)

##tarsila_asr = tarsila_asr.remove_columns(["origin", "duration", "gender"])

print(tarsila_asr)

print("End Load Dataset", datetime.datetime.now())


from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)

from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained(base_model, language="Portuguese", task="transcribe")

from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(base_model, language="Portuguese", task="transcribe")

def prepare_dataset(batch):
    batch["input_length"] = []
    batch["input_features"] = []
    batch["labels"] = []
    batch["labels_length"] = []

    for i in range(0, len(batch['audio'])):
        audio_info = batch['audio'][i]
        # compute input length
        batch["input_length"].append(len(audio_info["array"]))
        # compute log-Mel input features from input audio array
        batch["input_features"].append(feature_extractor(audio_info["array"], sampling_rate=audio_info["sampling_rate"]).input_features[0])
        # encode target text to label ids
        batch["labels"].append(tokenizer(batch["text"][i]).input_ids)
        # compute labels length
        batch["labels_length"].append(len(tokenizer(batch["text"][i], add_special_tokens=False).input_ids))
    return batch

#tarsila_asr = tarsila_asr.select_columns(["audio", "text"])
#tarsila_asr = tarsila_asr.map(prepare_dataset, batch_size=100, num_proc=1, batched=True)
mapped_tarsila_asr = DatasetDict()

for split_name, dataset in tarsila_asr.items():
    mapped_dataset = dataset.map(
        prepare_dataset,
        batched=True,
        batch_size=100
    )
    mapped_tarsila_asr[split_name] = mapped_dataset

tarsila_asr = mapped_tarsila_asr

# dataset
MAX_DURATION_IN_SECONDS = 30.0
max_input_length = MAX_DURATION_IN_SECONDS * 16000

def filter_inputs(input_length):
    """Filter inputs with zero input length or longer than 30s"""
    return 0 < input_length < max_input_length

#max_label_length = model.config.max_length
max_label_length = 448

def filter_labels(labels_length):
    """Filter label sequences longer than max length (448)"""
    return labels_length < max_label_length
    
tarsila_asr = tarsila_asr.filter(filter_inputs, input_columns=["input_length"])
tarsila_asr = tarsila_asr.filter(filter_labels, input_columns=["labels_length"])

print("End Load Dataset", datetime.datetime.now())

from transformers import WhisperForConditionalGeneration
#model = WhisperForConditionalGeneration.from_pretrained(base_model)
model = WhisperForConditionalGeneration.from_pretrained(f"{base_output_dir}/{fine_tuned_model_name}/checkpoint-150000")

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


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=base_output_dir+"/"+fine_tuned_model_name,  # change to a repo name of your choice
    run_name=fine_tuned_model_name,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=3,  # increase by 2x for every 2x decrease in batch size
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=3000000,
    #gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=25000,
    eval_steps=25000,
    logging_steps=1000,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    #push_to_hub=True,
    ddp_find_unused_parameters=False,
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

