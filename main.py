import fitz
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import embed_image, embed_text, process_pdf
from llm_utils import create_vectordb, load_vector_db, save_vectordb, retrieve_doc, create_multimodal_query, initialize_llm, get_llm_output
import numpy as np
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "pdfs", "cataract.pdf")
vector_db_path = os.path.join(script_dir, "outputs", "vectorstore")


vector_db = load_vector_db(vector_db_path)

if vector_db is None:
    print("no vector_db found")
    print(" creating new vector_db")
    print(f"processing pdfs..{pdf_path}.\n")
    all_docs, all_embeddings, image_data_store = process_pdf(pdf_path)
    embeddings_array = np.array(all_embeddings)
    docs_with_embeddings = list(zip(all_docs, embeddings_array))
    print("docs with embeddings created")
    print("trying to create vector_db")

    vector_db = create_vectordb(docs_with_embeddings)
    print("vector_store created\n")

    print("trying to save db")
    save_vectordb(vector_db, vector_db_path)
    print("vector_db_saved")

else:
    
    print("vector db is loaded and ready to go ")

if vector_db:
    print("vectordb is ready properly")
    query = "what does the new proposed methodoly say"
    print(f"testing with query {query}")
    all_docs, all_embeddings, image_data_store = process_pdf(pdf_path)
    top_k_docs = retrieve_doc(query, vector_db)
    multimodal_query = create_multimodal_query(query, top_k_docs, image_data_store)
    print("llm query ready")
    llm = initialize_llm()
    print("LLM ACTIVATED")
    response = get_llm_output(llm, multimodal_query)
    print(response)

