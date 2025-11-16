import praw
import os
import json
import re
import csv
from dotenv import load_dotenv

load_dotenv()

# Configura con tus credenciales de la app
reddit = praw.Reddit(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    user_agent=os.getenv('USER_AGENT'),
    rate_limit = 300
)
# Crear csv si no existe
bddir = 'bd/reddit/to_label.csv'

levels = [0,0,0]

fields = {"posts":{}}

pattern = re.compile(".*\[deleted\].*")

print ("Procesando...")

if not os.path.exists(bddir):
    f = open(bddir,'w', encoding='utf-8')
    f.close()
else:
    with open(bddir,newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader,1):
            print(f"Procesando {row[0]}, {i}")
            #time.sleep(1)
            try:
                sub = reddit.submission(id=row[0])
                field = {}
                field["subreddit"] = sub.subreddit.display_name
                field["title"] = sub.title
                field["upvote_ratio"] = sub.upvote_ratio
                field["num_comments"] = sub.num_comments
                field["score"] = sub.score
                field["selftext"] = sub.selftext
                if pattern.match(field["selftext"]):
                    continue
                field["ragescore"] = int(row[3])
                

                comms = {}
                #sub.comment_sort = "top"
                for c in sub.comments.list():
                    if type(c) is praw.models.Comment and not c.distinguished and c.parent_id == "t3_"+sub.id:
                        cfield = {}
                        cfield["body"] = c.body
                        cfield["score"] = c.score
                        comms[c.id] = cfield
                
                field["comments"] = comms


                fields["posts"][row[0]] = field
                match row[3]:
                    case "0":
                        levels[0]+=1
                    case "1":
                        levels[1]+=1
                    case "2":
                        levels[2]+=1
            except praw.exceptions.PRAWException:
                print("ERROR AL CARGAR POST!")


print(f"Se han encontrado {levels[0]} casos sin rb, {levels[1]} que pueden ser rb y {levels[2]} casos que son rb.")
print("Guardando...")
with open('reddit_db.json','w') as f:
    json.dump(fields,f)
print("Guardado")

