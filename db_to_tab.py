import json
import pandas as pd
from sklearn.model_selection import train_test_split
import re

COMM_LIMIT=40

dcc = {}
with open("reddit_db.json", "r", encoding="utf-8") as f:
    j = json.load(f)
    dcc = j["posts"]
pattern = re.compile(".*(http|\[deleted\]).*")

data = {
    "post" : [],
    #"upvote_ratio" : [],
    #"score" : [],
    #"comments" : [],
    "ragescore" : []
}

for k in dcc:
    comms = ""
    comms = dcc[k]["title"]+" "+dcc[k]["selftext"]
    #data["post"].append(dcc[k]["title"]+".\n"+dcc[k]["selftext"])
    #data["upvote_ratio"].append(dcc[k]["upvote_ratio"])
    #data["score"].append(dcc[k]["score"])
    i=0
    for c in dcc[k]["comments"]:
        if i >= COMM_LIMIT:
            break
        bod = dcc[k]["comments"][c]["body"]
        if not pattern.match(bod) and len(bod) > 0:
            i+=1
            comms+=bod+" "
    #data["comments"].append(comms)
    data["post"].append(comms)
    data["ragescore"].append(dcc[k]["ragescore"])
with open("tab_db.json", "w", encoding="utf-8") as f:
    json.dump(data,f)



df = pd.DataFrame(data)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    df[['post']],
    df['ragescore'],
    test_size=0.1,
    stratify=df['ragescore']
)

X_train.to_pickle("X_train.pkl")
X_test.to_pickle("X_test.pkl")
y_test.to_pickle("y_test.pkl")
y_train.to_pickle("y_train.pkl")
