import os
import streamlit as st
import bibtexparser

from embedding import create_embedding
from global_var import chunk, overlap
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Chroma

PAPERS_DIR = "./papers"

def load_bibtex_citations():
    bibtex_file = os.path.join(PAPERS_DIR, 'citations.bib')
    with open(bibtex_file, 'r') as f:
        bibtex_str = f.read()

    bib_database = bibtexparser.loads(bibtex_str)
    return {entry['ID']: entry for entry in bib_database.entries}

def open_pdf_at_page(pdf_path, page_number):
    # Enclose the path in quotes to handle spaces and special characters
    safe_path = f'"{pdf_path}"'

    # Ensure the page number is a string
    page_str = str(page_number)

    # Form the command string
    command = f"evince --page-label={page_str} {safe_path}.pdf"

    # Execute the command
    os.system(command)

def create_chroma_db():
    python_splitter = PythonCodeTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
    embeddings = create_embedding()
    return Chroma(persist_directory='db', embedding_function=embeddings)

def display_search_results(search_results, bibtex_citations):
    for index, result in enumerate(search_results):
        paper_name = result[0].metadata['paper_name']
        page_num = result[0].metadata['page_num']
        unique_button_key = f"{paper_name}_{page_num}_{index}"

        st.write(f"Paper Name: {paper_name}")
        st.write(f"Page Number: {page_num}")
        st.write(f"Content: {result[0].page_content}")
        st.write(f"Relevance: {result[1]}")

        if st.button(f'Open {paper_name} at page {page_num}', key=unique_button_key):
            open_pdf_at_page(os.path.join(PAPERS_DIR, paper_name), page_num)

        display_bibtex_citation(bibtex_citations, paper_name, page_num, unique_button_key)

        st.write('------------------')

def display_bibtex_citation(bibtex_citations, paper_name, page_num, unique_button_key):
    bibtex_entry = bibtex_citations.get(paper_name, None)
    if bibtex_entry:
        from bibtexparser.bibdatabase import BibDatabase

        bib_database = BibDatabase()
        bib_database.entries = [bibtex_entry]
        bibtex_text = bibtexparser.dumps(bib_database)
        cite_command = f"\\cite[p.~{page_num}]{paper_name}"

        st.write(f"LaTeX Citation: {cite_command}")


import pyperclip

def convert_ready_for_send(prompt, search_results, result_gpt3):
    string_to_send = ""
    string_to_send += "Frage: "
    string_to_send += str(prompt)
    string_to_send += "MATERIAL START: "
    string_to_send += str(result_gpt3)
    string_to_send += "MATERIAL END: "
    index = 0
    for result in search_results:
        string_to_send += "\nMATERIAL " + str(index) +  "START: \n" 
        string_to_send += str(result[0].page_content)
        string_to_send += "\nMATERIAL" + str(index) + " END\n"
        index += 1
    return string_to_send


from openai import OpenAI
# Initialisieren des GPT-3-Modells
client = OpenAI()

def generate_response(text, Instruction):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": text}
        ]
    )

    # Extrahieren Sie den Text der Antwort auf die korrekte Weise
    # Hier wird angenommen, dass 'message' ein Attribut des Objekts ist
    response_text = completion.choices[0].message.content
    return response_text

# import pdb
import streamlit as st

def main():
    st.title('Semantic Search with OpenAI and Streamlit')

    # Initialize session state variables if they don't exist
    if 'search_results' not in st.session_state:
        st.session_state['search_results'] = []
    if 'bibtex_citations' not in st.session_state:
        st.session_state['bibtex_citations'] = load_bibtex_citations()
    if 'db' not in st.session_state:
        st.session_state['db'] = create_chroma_db()

    prompt = st.text_input('Enter your search query')

    if st.button('Generate'):
        result_gpt = generate_response(prompt, "")
        search_results = st.session_state['db'].similarity_search_with_score(result_gpt, k=3)
        search_results.sort(key=lambda x: x[1], reverse=True)
        st.session_state['search_results'] = search_results
        pyperclip.copy(convert_ready_for_send(prompt, search_results, result_gpt))

    # Display search results if they exist
    if st.session_state['search_results']:
        display_search_results(st.session_state['search_results'], st.session_state['bibtex_citations'])

if __name__ == "__main__":
    main()
