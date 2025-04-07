from django.shortcuts import render
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

# Create your views here.
# Configura tu API Key de OpenAI
# Load environment variables from the .env file
# Cargar la API Key de OpenAI
load_dotenv('../api_keys.env')
client = OpenAI(api_key=os.environ.get('openai_apikey'))

# Función para calcular similitud de coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommend_movie(request):
    recommended_movie = None
    max_similarity = -1

    if request.method == "POST":
        # Recibir el prompt del usuario desde el formulario
        prompt = request.POST.get("prompt")

        # Generar embedding del prompt
        response = client.embeddings.create(
            input=[prompt],
            model="text-embedding-3-small"
        )
        prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

        # Recorrer la base de datos y calcular similitudes
        for movie in Movie.objects.all():
            movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
            similarity = cosine_similarity(prompt_emb, movie_emb)

            if similarity > max_similarity:
                max_similarity = similarity
                recommended_movie = movie

    return render(request, "recommend_movie.html", {
        "recommended_movie": recommended_movie,
        "similarity": max_similarity,
    })