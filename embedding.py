import torch 
from langchain.embeddings import HuggingFaceEmbeddings
def create_embedding():
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#    if torch.cuda.is_available():
#        embeddings.model = embeddings.model.to('cuda')
    return embeddings
