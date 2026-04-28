from pyfacebook import GraphAPI
import random

# Usa un token generado desde Graph API Explorer o Facebook App
ACCESS_TOKEN = "TU_TOKEN_DE_ACCESO"

# Crear la instancia de la API
api = GraphAPI(access_token=ACCESS_TOKEN)

# Obtener últimos posts de una página (por nombre o ID)
page_name = "nasa"
response = api.get_page_posts(page_id=page_name, count=50)

# Escoger uno al azar
posts = response.data
if posts:
    post = random.choice(posts)
    print("Post ID:", post.id)
    print("Mensaje:", post.message or "[Sin mensaje]")
else:
    print("No hay posts disponibles.")