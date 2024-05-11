from multiprocessing import Pool, cpu_count
import os
import pdfplumber
import bibtexparser
import json
import logging
from global_var import chunk, overlap  # Angenommen, diese sind in Ihrer ursprünglichen Datei definiert
from embedding import create_embedding
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Chroma

# Logging-Einstellungen
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

PAPERS_DIR = "./papers"

def get_bibtex_citations(bibtex_file):
    with open(bibtex_file, 'r') as f:
        bib_database = bibtexparser.load(f)
    return {entry['ID']: entry for entry in bib_database.entries}

bibtex_citations = get_bibtex_citations(os.path.join(PAPERS_DIR, 'citations.bib'))

def extract_text_from_pdf_with_metadata(pdf_path, bibtex_citations):
    paper_name = os.path.basename(pdf_path).replace('.pdf', '')
    if not paper_name:
        logging.error(f"Failed to extract paper name from {pdf_path}")
        return []
    with pdfplumber.open(pdf_path) as pdf:
        chunks_with_metadata = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            bibtex_citation = bibtex_citations.get(paper_name, {})
            if not bibtex_citation:
                logging.warning(f"No BibTeX citation found for {paper_name}")
            # Page Number wird ausgerechnet und dann Max von 0 und Page Number. 
            page_num = page_num - 1
            page_num = max(0, page_num)
            metadata = {
                'paper_name': paper_name,
                'page_num': page_num,
                'full_path': pdf_path,
                'bibtex_citation': bibtex_citation
            }
            chunks_with_metadata.append((text, metadata))
    return chunks_with_metadata

def validate_metadata(metadata):
    validated_metadata = {}
    for key, value in metadata.items():
        if key in ['paper_name', 'page_num', 'bibtex_citation'] and not value:
            logging.error(f"Missing or empty {key} in metadata: {metadata}")
            continue
        if isinstance(value, (str, int, float)):
            validated_metadata[key] = value
        elif isinstance(value, dict):
            validated_metadata[key] = json.dumps(value)
        else:
            logging.error(f"Unsupported metadata type for key {key}: {type(value)}")
    return validated_metadata

def process_file(pdf_file):
    pdf_path = os.path.join(PAPERS_DIR, pdf_file)
    texts_with_metadata = extract_text_from_pdf_with_metadata(pdf_path, bibtex_citations)
    validated_texts_with_metadata = [(text, validate_metadata(metadata)) for text, metadata in texts_with_metadata]
    return validated_texts_with_metadata

if __name__ == '__main__':
    num_cpus = cpu_count()
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]
    
    with Pool(processes=num_cpus) as pool:
        results = pool.map(process_file, pdf_files)
        
    all_texts_with_metadata = [item for sublist in results for item in sublist]
    
    # Split text into chunks with metadata
    python_splitter = PythonCodeTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
    docs = []
    for text, metadata in all_texts_with_metadata:
        validated_metadata = validate_metadata(metadata)  # Validate metadata
        chunks = python_splitter.create_documents([text])
        for chunk in chunks:
            chunk.metadata = validated_metadata  # Add validated metadata to each chunk
            docs.append(chunk)

    # Create embeddings
    embeddings = create_embedding()

    # Store the vectors
    store = Chroma.from_documents(docs, embeddings, persist_directory='db')
    store.persist()

