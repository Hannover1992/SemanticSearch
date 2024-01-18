
import os
import pdfplumber
import bibtexparser
import json  # Import json for serialization
from global_var import chunk, overlap
from embedding import create_embedding
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings
import torch

def create_embedding():
    # Initialisieren der Einbettungen mit Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Überprüfen, ob eine GPU verfügbar ist
    if torch.cuda.is_available():
        # Vorausgesetzt, embeddings.model ist das zugrunde liegende Modell
        embeddings.model.to('cuda')
    
    return embeddings

def get_embedding(text):
    embeddings = create_embedding()
    return embeddings.encode(text)

# Testen Sie die Funktion mit einem Beispieltext
if __name__ == "__main__":
    example_text = "Das ist ein Testtext."
    embedding_vector = get_embedding(example_text)
    print(embedding_vector)
