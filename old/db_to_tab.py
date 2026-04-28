import json
import pandas as pd
from sklearn.model_selection import train_test_split
import re

MAX_CHAR=2000

def normalize_text(text):
        text = re.sub(r'http\S+', '[URL]', text)
        text = re.sub(r'@\w+', '[USER]', text)
        text = re.sub(r'\!\[gif\]\(.*\)', '[URL]', text)
        return text

dcc = {}
with open("reddit_db.json", "r", encoding="utf-8") as f:
    j = json.load(f)
    dcc = j["posts"]
pattern = re.compile(".*\[deleted\].*")

data = {
    "post" : [],
    "upvote_ratio" : [],
    "score" : [],
    #"comments" : [],
    "ragescore" : []
}

for k in dcc:
    #comms = ""
    comms = dcc[k]["title"]+" "+dcc[k]["selftext"]
    for c in dcc[k]["comments"]:
        bod = dcc[k]["comments"][c]["body"]
        #if len(comms+bod+" ") > MAX_CHAR:
        #    break
        if not pattern.match(bod) and len(bod) > 0:
            comms+=bod+" "
    comms = normalize_text(comms)
    #data["comments"].append(comms)
    data["upvote_ratio"].append(dcc[k]["upvote_ratio"])
    data["score"].append(dcc[k]["score"])
    #data["post"].append(dcc[k]["title"]+".\n"+dcc[k]["selftext"])
    data["post"].append(comms)

    data["ragescore"].append(dcc[k]["ragescore"])
with open("tab_db.json", "w", encoding="utf-8") as f:
    json.dump(data,f)



df = pd.DataFrame(data)
df.to_pickle("all_df.pkl")
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    df[['post','score','upvote_ratio']],
    df['ragescore'],
    test_size=0.1,
    stratify=df['ragescore']
)

X_train.to_pickle("X_train.pkl")
X_test.to_pickle("X_test.pkl")
y_test.to_pickle("y_test.pkl")
y_train.to_pickle("y_train.pkl")
