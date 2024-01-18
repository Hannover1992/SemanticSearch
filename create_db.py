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
            metadata = {
                'paper_name': paper_name,
                'page_num': page_num + 1,
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
# Semantic Search Engine for Academic Papers

This project is a Semantic Search Engine for Academic Papers. It extracts text and metadata from PDF files of academic papers, validates the metadata, splits the text into chunks, and creates embeddings for each chunk. The embeddings are then stored in a database for semantic search.

## How to Run

1. Ensure that all the required libraries are installed. You can install them using pip:
```
pip install multiprocessing os pdfplumber bibtexparser json logging
```

2. Place your PDF files in the `./papers` directory.

3. Run the script:
```
python main.py
```

4. The script will process all PDF files in the `./papers` directory, extract text and metadata, validate the metadata, split the text into chunks, create embeddings for each chunk, and store the embeddings in a database.

5. You can then use the stored embeddings for semantic search.

## Note

This project uses multiprocessing to process multiple PDF files simultaneously. The number of processes is equal to the number of CPUs on your machine.

The project also uses logging to log errors and warnings. The log file is `app.log`.

The project assumes that the BibTeX citations for the papers are stored in a `.bib` file in the `./papers` directory. The BibTeX citations are used as part of the metadata for each chunk of text.

The project uses the `PythonCodeTextSplitter` class to split the text into chunks. The size of the chunks and the overlap between chunks can be configured in the `global_var.py` file.

The project uses the `Chroma` class to store the embeddings. The embeddings are stored in a directory named `db`.
