import json
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def normalize_text(text):
        text = re.sub(r'http\S+', '[URL]', text)
        text = re.sub(r'@\w+', '[USER]', text)
        return text

dcc = {}
with open("reddit_db.json", "r", encoding="utf-8") as f:
    j = json.load(f)
    dcc = j["posts"]
pattern = re.compile(".*\[deleted\].*")

f_dcc = {}


for k in dcc:
   pass

with open("e_reddit_db.json", "w", encoding="utf-8") as f:
    json.dump(f_dcc,f)
