import os

import regex as re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
nltk.download("punkt_tab")


def cleaning(text):
    clean = re.sub(r'\[\s*\d+\s*\]', '', text)
    clean = re.sub(r'\(\d+\)', '', clean)
    clean = re.sub(r'\n+', '\n', clean)
    clean = re.sub(r'\s+', ' ', clean)
    clean = clean.replace('Image gallery', '')

    clean = clean.strip()
    return clean

def making_chunks(text):
    sentences = sent_tokenize(text)

    chunks = []
    start = 0
    while start < len(sentences):
        end = start + 4
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start = end - 1
    return chunks


def get_data(folder_path):
    complete_data = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename),'r',encoding='utf-8') as f:
            raw_text = f.read()

        cleaned = cleaning(raw_text)

        chunked = making_chunks(cleaned)

        for i, chunks in enumerate(chunked,1):
            complete_data.append({
                'filename' : filename.strip('.txt'),
                'chunk_number' : i,
                'text' : chunks
            })
    return complete_data


