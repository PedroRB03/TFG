import random
import praw
import os
import csv
from dotenv import load_dotenv

load_dotenv()

# Configura con tus credenciales de la app
reddit = praw.Reddit(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    user_agent=os.getenv('USER_AGENT')
)
# Crear csv si no existe
bddir = 'bd/reddit/rb_tocheck.csv'

if not os.path.exists(bddir):
    f = open(bddir,'w', encoding='utf-8')
    f.close()


skip = set() # conjunto de existentes para skipear

with open(bddir,newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        skip.add(row[0]) # metemos id en el set para evitar repetir


# Buscamos ids de posts y comentarios interesantes entrantes

hotwords = ["rage-bait","rage bait","ragebait"]
found = []
limit = 300

print("Buscando...")
try:
    for c in reddit.subreddit("all").stream.comments():
        if c.submission.id not in skip and any(w in c.body.lower() for w in hotwords):
            found.append([c.submission.id, c.id,"https://www.reddit.com/"+c.permalink]) # metemos para guardar
            skip.add(c.submission.id) # evitamos repeticiones del mismo post
            print(f"[{c.subreddit.display_name}] sub: {c.submission.id}, com: {c.id} - https://www.reddit.com/{c.permalink}")
        if len(found) >= limit:
            break
except KeyboardInterrupt:
    print(f"Terminado a mano con {len(found)} casos de un total de {len(skip)}.")

print("Guardando...")
with open(bddir,'a',newline='', encoding='utf-8') as f: # estructura: id sub, id com, url
    #f.write('\n')
    w = csv.writer(f,)
    w.writerows(found)

print("Guardado terminado.")