from datasets import load_dataset
from tqdm.auto import tqdm
import csv
import re
import os

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
        with open(index_file, mode="w", encoding="utf-8") as f:
            f.write("idx\torigin\tduration\tgender\tref\tref_norm\n")
            for i in tqdm(range(len(dataset))):
                origin = dataset[i]['origin']
                duration = dataset[i]['duration']
                gender = dataset[i]['gender']
                ref = dataset[i]['text']
                ref_norm = replace_special_tokens_and_normalize(ref)

                f.write(f"{i}\t{origin}\t{duration}\t{gender}\t{ref}\t{ref_norm}\n")


def eval():
    print("loading dataset...")
    dataset_id = "sidleal/TARSILA-ASR-TST"
    dataset = load_dataset(dataset_id)['test']
    print(dataset)

    print("saving index file...")
    index_file = "tarsila-asr-index.csv"
    save_index_file(index_file, dataset)


    with open(index_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            print(row)

    


if __name__ == '__main__':
    eval()
