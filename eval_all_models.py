from datasets import load_dataset
from tqdm.auto import tqdm
import csv
import re
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from fairseq2.assets import AssetCard
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import io
import soundfile as sf
import numpy as np
from contextlib import ExitStack

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_list = [
    "facebookresearch__omniASR_LLM_1B",
    "facebookresearch__omniASR_LLM_300M",
    "openai__whisper-large-v3",
    "openai__whisper-medium",
    "sidleal__distil-whisper-coraa-mupe-asr-2",
    "sidleal__distil-whisper-tarsila-asr-v1-200k",
    "sidleal__distil-whisper-tarsila-asr-v1-750k",
    "sidleal__omniASR_LLM_1B_Tarsila_15k",
    "sidleal__omniASR_LLM_1B_Tarsila_4k",
    "sidleal__omniASR_LLM_1B_Tarsila_9k",
    "sidleal__omniASR_LLM_300M_Tarsila_4k",
    "sidleal__omniASR_LLM_300M_Tarsila_9k",
    "sidleal__whisper-tarsila-asr-large3-v1-450k",
    "sidleal__whisper-tarsila-asr-large3-v1-75k",
    "sidleal__whisper-tarsila-asr-medium-v1-100k",
    "sidleal__whisper-tarsila-asr-medium-v1-350k",
]


alphabet = r"ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõũúû1234567890%\-\n/\\ "

def replace_special_tokens_and_normalize(text):
    text = text.lower()

    map_words = {
        "éh": "eh",
        "ehm": "eh",
        "ehn": "eh",
        "hum": "uh",
        "hm": "uh",
        "uhm": "uh",
        "hã": "ah",
        "ãh": "ah",
        "ã":  "ah",
        "hmm": "uh",
        "mm": "uh",
        "mhm": "uh"
    }

    text = re.sub("h+", "h", text)
    text = re.sub("[^{}]".format(alphabet+" "), " ", text)
    text = re.sub("[ ]+", " ", text)

    words = text.split(' ')
    new_words = []
    for word in words:
        if word == '' or word == ' ':
            continue
        if word in map_words:
            new_words.append(map_words[word])
        else:
            new_words.append(word)

    return " ".join(new_words)


def save_index_file(index_file, dataset):
    exists = os.path.isfile(index_file)
    if not exists:
        with open(index_file, mode="w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(["idx", "origin", "duration", "gender", "ref", "ref_norm"])
            
            for i in tqdm(range(len(dataset))):
                ref = dataset[i]['text'].replace('\n', ' ')
                ref_norm = replace_special_tokens_and_normalize(ref)
                
                writer.writerow([
                    i, 
                    dataset[i]['origin'], 
                    dataset[i]['duration'], 
                    dataset[i]['gender'], 
                    ref, 
                    ref_norm
                ])

def eval_whisper_based(model_id, dataset):
    print(f"eval whisper based model: {model_id}...")
    model_csv_file = f"{model_id}.csv".replace("/", "__")
    exists = os.path.isfile(model_csv_file)
    if exists:
        print("skipping...")
        return

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        dtype=torch_dtype,
        device=device,
    )

    with open(model_csv_file, mode="w", encoding="utf-8", newline='') as fm:
        writer = csv.writer(fm)
        writer.writerow(["idx", "out", "out_norm", "time_in_ms"])

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for i in tqdm(range(len(dataset))):
            try:
                start_event.record()
                out = pipe(dataset[i]["audio"])
                end_event.record()

                torch.cuda.synchronize()
                time_in_ms = start_event.elapsed_time(end_event)

                out_text = out['text']
                out_norm = replace_special_tokens_and_normalize(out_text)
                writer.writerow([
                    i,
                    out_text,
                    out_norm,
                    round(time_in_ms, 2)
                ])
            except Exception as e:
                print(f"Skipping index {i} due to error: {e}")
                writer.writerow([
                    i, 
                    "error", 
                    "error",
                    -1
                ])
            if i > 5:
                break
    return


def eval_omni_based(model_id, dataset):
    print(f"eval whisper based model: {model_id}...")
    model_csv_file = f"{model_id}.csv".replace("/", "__")
    exists = os.path.isfile(model_csv_file)
    if exists:
        print("skipping...")
        return

    model_card = ''
    if model_id == "sidleal/omniASR_LLM_300M_Tarsila_4k":
        model_card = AssetCard("omniASR_LLM_300M_Tarsila_4k", {
            "model_family": "wav2vec2_llama", 
            "model_arch": "300m",   
            "checkpoint": "https://huggingface.co/sidleal/omniASR_LLM_300M_Tarsila_4k/resolve/main/sdp_00.pt?download=true",
            "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
            "tokenizer_family": "char_tokenizer",
        })

    elif model_id == "sidleal/omniASR_LLM_300M_Tarsila_9k":
        model_card = AssetCard("omniASR_LLM_300M_Tarsila_9k", {
            "model_family": "wav2vec2_llama", 
            "model_arch": "300m",   
            "checkpoint": "https://huggingface.co/sidleal/omniASR_LLM_300M_Tarsila_9k/resolve/main/sdp_00.pt?download=true",
            "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
            "tokenizer_family": "char_tokenizer",
        })

    elif model_id == "sidleal/omniASR_LLM_1B_Tarsila_4k":
        model_card = AssetCard("omniASR_LLM_1B_Tarsila_4k", {
            "model_family": "wav2vec2_llama", 
            "model_arch": "1b",   
            "checkpoint": "https://huggingface.co/sidleal/omniASR_LLM_1B_Tarsila_4k/resolve/main/sdp_00.pt?download=true",
            "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
            "tokenizer_family": "char_tokenizer",
        })

    elif model_id == "sidleal/omniASR_LLM_1B_Tarsila_9k":
        model_card = AssetCard("omniASR_LLM_1B_Tarsila_9k", {
            "model_family": "wav2vec2_llama", 
            "model_arch": "1b",   
            "checkpoint": "https://huggingface.co/sidleal/omniASR_LLM_1B_Tarsila_9k/resolve/main/sdp_00.pt?download=true",
            "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
            "tokenizer_family": "char_tokenizer",
        })

    elif model_id == "sidleal/omniASR_LLM_1B_Tarsila_15k":
        model_card = AssetCard("omniASR_LLM_1B_Tarsila_15k", {
            "model_family": "wav2vec2_llama", 
            "model_arch": "1b",   
            "checkpoint": "https://huggingface.co/sidleal/omniASR_LLM_1B_Tarsila_15k/resolve/main/sdp_00.pt?download=true",
            "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
            "tokenizer_family": "char_tokenizer",
        })

    elif model_id == "facebookresearch/omniASR_LLM_7B":
        model_card = AssetCard("omniASR_LLM_7B", {
            "model_family": "wav2vec2_llama", 
            "model_arch": "7b_v1_variant7",   
            "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B.pt",
            "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_v7.model",
            "tokenizer_family": "char_tokenizer",
        })

    elif model_id == "facebookresearch/omniASR_LLM_1B":
        model_card = AssetCard("omniASR_LLM_1B", {
            "model_family": "wav2vec2_llama", 
            "model_arch": "1b",   
            "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-1B.pt",
            "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
            "tokenizer_family": "char_tokenizer",
        })

    elif model_id == "facebookresearch/omniASR_LLM_300M":
        model_card = AssetCard("omniASR_LLM_300M", {
            "model_family": "wav2vec2_llama", 
            "model_arch": "300m",   
            "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-300M.pt",
            "tokenizer": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
            "tokenizer_family": "char_tokenizer",
        })

    else:
        print(f"invalid model: {model_id}")
        return

    pipeline = ASRInferencePipeline(model_card=model_card, device=device, dtype=torch_dtype)

    sample_rate = 16000
    lang = ["por_Latn"]

    with open(model_csv_file, mode="w", encoding="utf-8", newline='') as fm:
        writer = csv.writer(fm)
        writer.writerow(["idx", "out", "out_norm", "time_in_ms"])

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for i in tqdm(range(len(dataset))):
            try:
                audio_array = dataset[i]["audio"]["array"]
                buf = io.BytesIO()
                sf.write(buf, audio_array, sample_rate, format='WAV')
                buf.seek(0)
                raw_uint8_data = np.frombuffer(buf.read(), dtype=np.uint8)                
                audio_files = [raw_uint8_data]

                start_event.record()
                transcriptions = pipeline.transcribe(audio_files, lang=lang, batch_size=1)                
                end_event.record()

                torch.cuda.synchronize()
                time_in_ms = start_event.elapsed_time(end_event)

                out_text = transcriptions[0]
                out_norm = replace_special_tokens_and_normalize(out_text)
                writer.writerow([
                    i,
                    out_text,
                    out_norm,
                    round(time_in_ms, 2)
                ])
            except Exception as e:
                print(f"Skipping index {i} due to error: {e}")
                writer.writerow([
                    i, 
                    "error", 
                    "error",
                    -1
                ])
            if i > 5:
                break #=====================================


def merge_results(index_file):
    print("merging results...")
    merged_csv_file = "tarsila-asr-results-merged.csv"
    exists = os.path.isfile(merged_csv_file)
    if exists:
        return merged_csv_file

    with ExitStack() as stack:
        main_file = stack.enter_context(open(index_file, 'r', encoding='utf-8'))
        model_files = [stack.enter_context(open(f"{m}.csv", 'r', encoding='utf-8')) for m in model_list]
        out_file = stack.enter_context(open(merged_csv_file, 'w', encoding='utf-8', newline=''))

        main_reader = csv.DictReader(main_file)
        model_readers = [csv.DictReader(f) for f in model_files]
        
        base_headers = ["idx", "origin", "duration", "gender", "ref", "ref_norm"]
        dynamic_headers = []
        for m in model_list:
            model_name = m.split("__")[1]
            dynamic_headers.extend([f"{model_name}_out", f"{model_name}_out_norm", f"{model_name}_time_in_ms"])
        
        writer = csv.DictWriter(out_file, fieldnames=base_headers + dynamic_headers)
        writer.writeheader()

        for main_row, *model_rows in zip(main_reader, *model_readers):
            new_row = main_row.copy()
            
            for i, m_row in enumerate(model_rows):
                model_name = model_list[i].split("__")[1]
                new_row[f"{model_name}_out"] = m_row.get('out', '')
                new_row[f"{model_name}_out_norm"] = m_row.get('out_norm', '')
                new_row[f"{model_name}_time_in_ms"] = m_row.get('time_in_ms', '')
            
            writer.writerow(new_row)

    return merged_csv_file


def calc_metrics(merged_file):
    print("calc wer cer rtf...")
    metrics_csv_file = "tarsila-asr-results-merged-metrics.csv"
    exists = os.path.isfile(metrics_csv_file)
    if exists:
        return

    with open(metrics_csv_file, mode="w", encoding="utf-8", newline='') as fm:
        with open(merged_file, 'r') as f:
            reader = csv.DictReader(f)

            base_headers = reader.fieldnames
            dynamic_headers = []
            for m in model_list:
                model_name = m.split("__")[1]
                dynamic_headers.extend([f"{model_name}_wer", f"{model_name}_cer", f"{model_name}_rtf"])
            writer = csv.DictWriter(fm, fieldnames=base_headers + dynamic_headers)
            writer.writeheader()

            for row in reader:
                new_row = row.copy()
                
                for m in model_list:
                    model_name = m.split("__")[1]
                    new_row[f"{model_name}_wer"] = 0
                    new_row[f"{model_name}_cer"] = 0
                    new_row[f"{model_name}_rtf"] = row.get('time_in_ms', '-1')
                
                writer.writerow(new_row)


def eval():
    print("loading dataset...")
    dataset_id = "sidleal/TARSILA-ASR-TST"
    dataset = load_dataset(dataset_id)['test']
    print(dataset)

    print("saving index file...")
    index_file = "tarsila-asr-index.csv"
    save_index_file(index_file, dataset)

    eval_whisper_based("sidleal/distil-whisper-coraa-mupe-asr-2", dataset)
    eval_whisper_based("sidleal/distil-whisper-tarsila-asr-v1-200k", dataset)
    eval_whisper_based("sidleal/distil-whisper-tarsila-asr-v1-750k", dataset)
    
    eval_whisper_based("openai/whisper-medium", dataset)
    eval_whisper_based("openai/whisper-large-v3", dataset)

    eval_whisper_based("sidleal/whisper-tarsila-asr-medium-v1-100k", dataset)
    eval_whisper_based("sidleal/whisper-tarsila-asr-medium-v1-350k", dataset)
    eval_whisper_based("sidleal/whisper-tarsila-asr-large3-v1-75k", dataset)
    eval_whisper_based("sidleal/whisper-tarsila-asr-large3-v1-450k", dataset)

    eval_omni_based("sidleal/omniASR_LLM_300M_Tarsila_4k", dataset)
    eval_omni_based("sidleal/omniASR_LLM_300M_Tarsila_9k", dataset)
    eval_omni_based("sidleal/omniASR_LLM_1B_Tarsila_4k", dataset)
    eval_omni_based("sidleal/omniASR_LLM_1B_Tarsila_9k", dataset)
    eval_omni_based("sidleal/omniASR_LLM_1B_Tarsila_15k", dataset)

    eval_omni_based("facebookresearch/omniASR_LLM_300M", dataset)
    eval_omni_based("facebookresearch/omniASR_LLM_1B", dataset)
    #eval_omni_based("facebookresearch/omniASR_LLM_7B", dataset)

    merged_file = merge_results(index_file)        

    calc_metrics(merged_file)


if __name__ == '__main__':
    eval()
