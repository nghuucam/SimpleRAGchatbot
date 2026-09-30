import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Đây là model embedding MIỄN PHÍ của Google:
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector = embeddings.embed_query("hello, world!")
print(vector[:5])