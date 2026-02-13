def train_model():
    import os
    import torch
    import evaluate
    from typing import Any, Dict, List, Union
    from dataclasses import dataclass
    from transformers import Seq2SeqTrainingArguments
    from transformers import Seq2SeqTrainer
    from transformers import WhisperForConditionalGeneration
    from transformers import WhisperFeatureExtractor
    from transformers import WhisperTokenizer
    from transformers import WhisperProcessor
    from datasets import load_dataset, DatasetDict
    import wandb

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # start group training
    #torch.distributed.init_process_group()

    instance_num = os.environ.get("LOCAL_RANK", -1)
    # configs
    os.environ["WANDB_API_KEY"] = "xxxx" 
    os.environ["WANDB_PROJECT"] = "tarsila-asr"
    os.environ["WANDB_RUN_NAME"] = f"tarsila-asr-run-17-{instance_num}"
    wandb.init(project="tarsila-asr", name=f"tarsila-asr-run-17-{instance_num}")

    hf_token = "xxxx"
    base_output_dir = "/home/jovyan/training"
    fine_tuned_model_name = "distil-whisper-tarsila-asr-v1"
    distilled_model_link = "nilc-nlp/distil-whisper-coraa-mupe-asr"
    dataset_link = "sidleal/TARSILA-ASR-V1"

    # model
    #model = WhisperForConditionalGeneration.from_pretrained(distilled_model_link)
    model = WhisperForConditionalGeneration.from_pretrained(f"{base_output_dir}/{fine_tuned_model_name}/checkpoint-200000")
    model.generation_config.language = "portuguese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

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

    feature_extractor = WhisperFeatureExtractor.from_pretrained(distilled_model_link)
    tokenizer = WhisperTokenizer.from_pretrained(distilled_model_link, language="Portuguese", task="transcribe")
    processor = WhisperProcessor.from_pretrained(distilled_model_link, language="Portuguese", task="transcribe")

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

    tarsila_asr = DatasetDict()
    tarsila_asr["train"] = load_dataset(dataset_link, split="train", token=hf_token)
    tarsila_asr["validation"] = load_dataset(dataset_link, split="validation", token=hf_token)

    #tests
    #tarsila_asr["train"] = tarsila_asr["train"].rename_column("original_text", "text")
    #tarsila_asr["validation"] = tarsila_asr["validation"].rename_column("original_text", "text")
    #tarsila_asr["train"] = tarsila_asr["train"].select(range(1000))
    #tarsila_asr["validation"] = tarsila_asr["validation"].select(range(1000))

    tarsila_asr_2 = tarsila_asr.select_columns(["audio", "text"])

    tarsila_asr_3 = tarsila_asr_2.map(prepare_dataset, batch_size=100, num_proc=1, batched=True)

    #print(tarsila_asr_3)
    #print(tarsila_asr_3['train'][2323]['labels_length'])
    #print(tarsila_asr_3['validation'][2313]['labels_length'])

    tarsila_asr_3 = tarsila_asr_3.filter(filter_inputs, input_columns=["input_length"])
    tarsila_asr_3 = tarsila_asr_3.filter(filter_labels, input_columns=["labels_length"])

    #tarsila_asr_3.push_to_hub("sidleal/TARSILA-ASR-V1-clean", private=True, token=hf_token, commit_message="v1 clean")

    # collector
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int
        #def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        #    # split inputs and labels since they have to be of different lengths and need different padding methods
        #    # first treat the audio inputs by simply returning torch tensors
        #    input_features = [{"input_features": feature["input_features"]} for feature in features]
        #    batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        #    # get the tokenized label sequences
        #    label_features = [{"input_ids": feature["labels"]} for feature in features]
        #    # pad the labels to max length
        #    labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        #    # replace padding with -100 to ignore loss correctly
        #    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        #    # if bos token is appended in previous tokenization step,
        #    # cut bos token here as it's append later anyways
        #    if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
        #        labels = labels[:, 1:]
        #    batch["labels"] = labels
        #    return batch
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # --- Audio Inputs ---
            # First, treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
    
            # --- Labels ---
            # Get the tokenized label sequences
            label_features = [torch.tensor(feature["labels"]) for feature in features]
            
            # Manually pad the labels to max length using torch.nn.utils.rnn.pad_sequence
            # We need to pad to the right, and the padding value should be -100
            labels = torch.nn.utils.rnn.pad_sequence(
                label_features,
                batch_first=True,
                padding_value=-100  # Set padding value to -100 directly
            )
    
            # Handle the BOS token.
            # Your original logic is correct here.
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
    
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # loss
    
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

    # train
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{base_output_dir}/" + fine_tuned_model_name, 
        run_name = fine_tuned_model_name,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        dataloader_num_workers=8, 
        dataloader_pin_memory=True,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=3000000,
        #gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=25000,
        eval_steps=25000,
        logging_steps=1000,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        #push_to_hub=True,
        #num_train_epochs=5,
        ##local_rank=os.environ.get("LOCAL_RANK", -1), # Used by Accelerate for DDP
        ##ddp_backend="nccl", # Recommended for NVIDIA GPUs 
        #hub_token=hf_token,
        ddp_find_unused_parameters=False, #standard finetuning
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=tarsila_asr_3["train"],
        eval_dataset=tarsila_asr_3["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )

    #trainer.train()

    kwargs = {
        "dataset_tags": "sidleal/TARSILA-ASR-V1",
        "dataset": "TARSILA-ASR-V1",  # a 'pretty' name for the training dataset
        "dataset_args": "split: train",
        "language": "pt",
        "model_name": "distil-whisper-tarsila-asr-v1-200k",  # a 'pretty' name for your model
        "finetuned_from": distilled_model_link,
        "tasks": "automatic-speech-recognition",
    }

    trainer.push_to_hub(**kwargs, token=hf_token, private=True)

    print(f"Train finished.")

    #torch.distributed.destroy_process_group()
    wandb.finish()

if __name__ == '__main__':
    train_model()