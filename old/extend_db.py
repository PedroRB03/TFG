import json
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def normalize_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\!\[gif\]\(.*\)', '', text)
        return text

def rem_o_start(text):
        text = re.sub(r'paraphrasedoutput:', '', text)
        return text

def elegir_mas_distinto(original, candidatos):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_orig = embedder.encode(
        original,
        convert_to_tensor=True,
        device=device
    )

    emb_cand = embedder.encode(
        candidatos,
        convert_to_tensor=True,
        device=device
    )

    similitudes = util.cos_sim(emb_orig, emb_cand)[0]

    idx = similitudes.argmin().cpu().item()
    return candidatos[idx]

def gen_reescritura(texto, max_length=128,n=5):
    prompt = f"paraphrase: {normalize_text(texto)} </s>"
    

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=10,
        num_return_sequences=n,
        do_sample=True,
        temperature = 1.5,
        top_p = 0.9,
        repetition_penalty = 1.8,
        no_repeat_ngram_size = 3
    )

    return [
        tokenizer.decode(o, skip_special_tokens=True)
        for o in outputs
    ]

def reescribir(texto,max_length=128,n=5):
    candidatos = gen_reescritura(texto,max_length,n)
    return rem_o_start(elegir_mas_distinto(texto, candidatos))

dcc = {}
with open("reddit_db.json", "r", encoding="utf-8") as f:
    j = json.load(f)
    dcc = j["posts"]
pattern = re.compile(".*\[(deleted|removed)\].*")



for k in dcc:
    print(f"Procesando: {k}")
    t = dcc[k]['title']
    st = dcc[k]['selftext']
    coms = {}
    for c in dcc[k]["comments"]:
        bod = dcc[k]["comments"][c]["body"]
        if not pattern.match(bod) and len(bod) > 0:
            coms[c] = {
                'body' : reescribir(bod,max_length=min(512,max(len(bod),64))),
                'score' : dcc[k]['comments'][c]['score']
            }
            print(bod + " [PR] :" + coms[c]['body'])
    dcc[k]['comments'] = coms
    if len(dcc[k]['title']) > 0:
        dcc[k]["title"] = reescribir(t,max_length=min(512,max(len(t),64)))
    if len(dcc[k]['selftext']) > 0:
        dcc[k]["selftext"] = reescribir(st,max_length=min(512,max(len(st),64)))


with open("e_reddit_db.json", "w", encoding="utf-8") as f:
    json.dump({'posts': dcc},f)
