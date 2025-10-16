import random
import praw
import os
from dotenv import load_dotenv

load_dotenv()

# Configura con tus credenciales de la app
reddit = praw.Reddit(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    user_agent=os.getenv('USER_AGENT')
)

def obtener_posts_aleatorios(subreddit_name, sample_size=500)->list:
    sub = reddit.subreddit(subreddit_name)
    # Obtener una muestra de submissions recientes (hot/new/top)
    # Ajusta sample_size según tus límites y rate limits
    submissions = list(sub.new(limit=sample_size))  # o .hot(), .top("week")
    if not submissions:
        return None
    random.shuffle(submissions)
    return submissions

def extraer_comentarios(submission, max_comments=None):
    submission.comments.replace_more(limit=None)  # obtiene todos los nodos "more"
    all_comments = submission.comments.list()
    if max_comments:
        all_comments = all_comments[:max_comments]
    # devuelve lista de dicts simples
    return [{"id": c.id, "author": str(c.author), "body": c.body, "score": c.score} for c in all_comments]

# Uso
subms = obtener_posts_aleatorios("python", sample_size=100)
if subms:
    for subm in subms:
        print("Título:", subm.title)
        comments = extraer_comentarios(subm, max_comments=200)
        print(f"Se obtuvieron {len(comments)} comentarios (máx 200)")
else:
    print("no se obtuvo sub")