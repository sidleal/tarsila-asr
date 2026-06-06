# Tarsila-ASR: A Multi-Domain Test Suite for Benchmarking Brazilian Portuguese Speech Recognition
## Test Dataset Link: [https://huggingface.co/datasets/nilc-nlp/TARSILA-ASR]

Citing this work:
-----------------
````
@article{TarsilaASR2026,
    author = {Sidney Evaldo Leal and Ariadne Matos and Edresson Casanova and Frederico Gonçalves and Renato Moraes Silva and Arnaldo Candido Jr and Sandra Maria Aluísio},
    title = {Tarsila-ASR: A Multi-Domain Test Suite for Benchmarking Brazilian Portuguese Speech Recognition},
    journal = {Proceedings of Interspeech 2026},
    year = {2026}
}
````

## Best Models WER vs RTF
This plot shows the Word Error Rate values as bars for each of our three primary models, and also the Real Time Factor values as lines. Although the best WER is achieved with the fine-tuned Whisper-Large, DistilWhisper provides the best tradeoff between accuracy and efficiency.

<img width="1489" height="790" alt="image" src="tarsila_subsets_wer_rtf2.png" />

## Folders description

| Folder | Content |
| :--- | :--- |
| tarsila-asr-dataset | Scripts for building the train/dev/test subsets from public datasets. |
| distil-whisper | Scripts for fine-tuning Distil-Whisper.  |
| whisper-large | Scripts for fine-tuning Whisper Large and Medium.  |
| voice-gender-classifier | Scripts for estimating gender.  |
| h100 | Outputs for eval_all_models.py running in the H100 env.  |
| rtx4070 | Outputs for eval_all_models.py running in the RTX 4070 env.  |

## Huggingface Tarsila-ASR dataset:
[https://huggingface.co/datasets/nilc-nlp/TARSILA-ASR]

## Huggingface Models checkpoints:
| Name | link |
| :--- | :--- |
| distil-whisper-ft-asr-200k | [https://huggingface.co/sidleal/distil-whisper-tarsila-asr-v1-200k] |
| distil-whisper-ft-asr-750k | [https://huggingface.co/sidleal/distil-whisper-tarsila-asr-v1-750k] |
| omniASR_LLM_1B_ft_15k | [https://huggingface.co/sidleal/omniASR_LLM_1B_Tarsila_15k] |
| omniASR_LLM_1B_ft_4k | [https://huggingface.co/sidleal/omniASR_LLM_1B_Tarsila_4k] |
| omniASR_LLM_1B_ft_9k | [https://huggingface.co/sidleal/omniASR_LLM_1B_Tarsila_9k] |
| omniASR_LLM_300M_ft_4k | [https://huggingface.co/sidleal/omniASR_LLM_300M_Tarsila_4k] |
| omniASR_LLM_300M_ft_9k | [https://huggingface.co/sidleal/omniASR_LLM_300M_Tarsila_9k] |
| whisper-ft-large3-v1-450k | [https://huggingface.co/sidleal/whisper-tarsila-asr-large3-v1-450k] |
| whisper-ft-large3-v1-75k | [https://huggingface.co/sidleal/whisper-tarsila-asr-large3-v1-75k] |
| whisper-ft-medium-v1-100k | [https://huggingface.co/sidleal/whisper-tarsila-asr-medium-v1-100k] |
| whisper-ft-medium-v1-350k | [https://huggingface.co/sidleal/whisper-tarsila-asr-medium-v1-350k] |

## Gender analysis
<img width="1480"  alt="tarsila_subsets_gender_wer" src="tarsila_subsets_gender_wer.png" />
