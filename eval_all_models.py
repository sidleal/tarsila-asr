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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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
            break #=====================================


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
    eval_omni_based("facebookresearch/omniASR_LLM_7B", dataset)

    # i = 0
    # with open(index_file, 'r') as f:
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         print(row['ref'])
    #         print(dataset[i]['text'])
    #         print('-')
    #         original = dataset[i]['text'].replace('\n', ' ')
    #         ref = row['ref']
    #         if ref != original:
    #             print(i, row['origin'])
    #             break
    #         i+=1
    


if __name__ == '__main__':
    eval()
