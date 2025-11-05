import praw
import os
import re
import time
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
bddir = 'bd/reddit/tocheck_improved.csv'
bdsubsdir = 'bd/reddit/checkedsubs.csv'

if not os.path.exists(bddir):
    f = open(bddir,'w', encoding='utf-8')
    f.close()


skip = set() # conjunto de existentes para skipear
subs = []
subsleft = []

with open(bddir,newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        skip.add(row[0]) # metemos id en el set para evitar repetir

with open(bdsubsdir,newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        subs.append(row[0]) 
        subsleft.append(row) 

# Buscamos ids de posts y comentarios interesantes entrantes

pattern = re.compile(".*bait.*")
#pattern = re.compile(".*[0-9]/[0-9] (ragebait|bait).*")

found = []
limit = 5000
fi = 0

def guardar(found):
    print("Guardando...")
    with open(bddir,'a',newline='', encoding='utf-8') as f: # estructura: id sub, id com, url
        w = csv.writer(f)
        w.writerows(found)
    print("Guardado terminado.")

def buscar_subs():
    found = []
    for subr in reddit.subreddits.popular(limit=5000):
        found.append([subr.display_name])
        print(len(found))
    with open(bdsubsdir,'a',newline='', encoding='utf-8') as f: # estructura: id sub, id com, url
        w = csv.writer(f)
        w.writerows(found)


print("Buscando...")
try:
    for c in reddit.subreddit("all").stream.comments():
        sub = c.submission
        if type(c) is praw.models.Comment and sub.id not in skip and not c.distinguished and c.parent_id == "t3_"+sub.id and pattern.match(c.body):
            found.append([sub.id, c.id,"https://www.reddit.com/"+c.permalink]) # metemos para guardar
            fi+=1
            skip.add(sub.id) # evitamos repeticiones del mismo post
            print(f"[{c.subreddit.display_name}] sub: {sub.id}, com: {c.id} - https://www.reddit.com/{c.permalink}")
            if len(found) >= 10:
                guardar(found)
                time.sleep(10)
                found = []
        if fi >= limit:
            break
        time.sleep(1)
except KeyboardInterrupt:
    print(f"Terminado a mano con {fi} casos de un total de {len(skip)}.")
except praw.exceptions.PRAWException:
    print("Terminado por TooManyRequests.")
#try:
#    for snam in subs:
#        subr = reddit.subreddit(snam)
#        print(f"sub {snam}")
#        for i,sub in enumerate(subr.new(limit=1000),1):
#            for c in sub.comments:
#                if c is praw.models.Comment and sub.id not in skip and not sub.distinguished and not c.distinguished and c.parent_id == "t3_"+sub.id and pattern.match(c.body):
#                    found.append([sub.id, c.id,"https://www.reddit.com/"+c.permalink]) # metemos para guardar
#                    skip.add(sub.id) # evitamos repeticiones del mismo post
#                    print(f"[{c.subreddit.display_name}] sub: {sub.id}, com: {c.id} - https://www.reddit.com/{c.permalink}")
#                if len(found) >= limit:
#                    break
#            if i%10 == 0:
#                print(f"subm {i}")
#            if len(found) >= limit:
#                break
#        if len(found) >= limit:
#            break
#        else:
#            subsleft.pop(0)
#except KeyboardInterrupt:
#    print(f"Terminado a mano con {len(found)} casos de un total de {len(skip)}.")
#except praw.exceptions.TooManyRequests:
#    print(f"Cortado, demasiadas requests")
guardar(found)