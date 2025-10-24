import basc_py4chan as fourchan
import requests
import random

# Obtenemos lista de boards
resp = requests.get("https://a.4cdn.org/boards.json")
resp.raise_for_status()
boards_data = resp.json()['boards']
board_names = [b['board'] for b in boards_data]

# Board aleatorio
board_name = random.choice(board_names)
print(board_name)
board = fourchan.Board(board_name)

# Paso 2: Obtener hilos actuales
threads = board.get_all_threads()
print(len(threads))
thread = random.choice(threads)

posts = thread.posts
print(len(posts))
#post = random.choice(posts)

# Paso 4: Mostrar info
#print(f"Board: /{board_name}/")
#print(f"Thread: {thread.id}")
#print("Post ID:", post.post_id)
#print("Comment:", post.comment)