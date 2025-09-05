import os  
import io
import base64
from pathlib import Path

import streamlit as st
from PIL import Image
import fitz 

from utils import process_pdf
from llm_utils import (
    create_vectordb,
    load_vector_db,
    save_vectordb,
    retrieve_doc,
)
import numpy as np
from langchain_google_genai import GoogleGenerativeAI


def extract_images_base64_only(pdf_path):
    image_data_store = {}
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_id = f"page_{i}_img_{img_index}"
                image_data_store[image_id] = img_base64
            except Exception:
                continue
    return image_data_store


def decode_base64_image(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def build_page_to_image_map(image_data_store):
    page_to_image_ids = {}
    for image_id in image_data_store.keys():
        try:
            # image_id format: "page_{i}_img_{j}"
            page_idx = int(image_id.split("_")[1])
        except Exception:
            continue
        page_to_image_ids.setdefault(page_idx, []).append(image_id)
    return page_to_image_ids


def build_context_text(docs):
    parts = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        doc_type = meta.get("type")
        page_num = meta.get("page")
        if doc_type == "text" and d.page_content:
            content = d.page_content.strip()
            if len(content) > 1200:
                content = content[:1200] + "..."
            parts.append(f"[page {page_num}] {content}")
        elif doc_type == "image":
            image_id = meta.get("image_id")
            parts.append(f"[page {page_num}] [Image: {image_id}]")
    return "\n\n".join(parts)


def render_pdf(pdf_file_path):
    try:
        with open(pdf_file_path, "rb") as f:
            pdf_bytes = f.read()
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = f"""
            <iframe
                src="data:application/pdf;base64,{b64_pdf}#view=FitH"
                width="100%"
                height="700px"
                type="application/pdf">
            </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.download_button("Download PDF", data=pdf_bytes, file_name=Path(pdf_file_path).name, mime="application/pdf")
    except Exception as e:
        st.warning(f"Unable to display PDF: {e}")


st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal RAG – Image-aware Retrieval")


if "llm" not in st.session_state:
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.session_state.llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key) if api_key else None

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "pdfs", "cataract.pdf")
vector_db_path = os.path.join(script_dir, "outputs", "vectorstore")

with st.sidebar:
    st.header("Index")
    st.write("PDF:", Path(pdf_path).name)
    top_k = st.slider("Top K", 1, 10, 5)
    if st.button("Clear chat"):
        st.session_state["messages"] = []
    show_pdf = st.checkbox("View PDF", value=False)

vector_db = load_vector_db(vector_db_path)
if vector_db is None:
    st.info("No vector DB found. Building index – this may take a few minutes…")
    all_docs, all_embeddings, image_data_store_full = process_pdf(pdf_path)
    embeddings_array = np.array(all_embeddings)
    docs_with_embeddings = list(zip(all_docs, embeddings_array))
    vector_db = create_vectordb(docs_with_embeddings)
    save_vectordb(vector_db, vector_db_path)
    st.success("Index created.")

if "image_data_store" not in st.session_state:
    st.session_state.image_data_store = extract_images_base64_only(pdf_path)
    st.session_state.page_to_image_ids = build_page_to_image_map(st.session_state.image_data_store)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about the PDF."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about the document")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if vector_db is None:
        err = "Vector DB not available. Please rebuild and try again."
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)
    else:
        retrieved_docs = retrieve_doc(user_input, vector_db, k=top_k)
        context_text = build_context_text(retrieved_docs)

        if st.session_state.llm is None:
            answer_text = "LLM is not configured. Please set GOOGLE_API_KEY in your environment."
        else:
            prompt = f"""Answer the question strictly using the context below. If the answer is not found, say you don't know.

<context>
{context_text}
</context>

Question: {user_input}"""
            try:
                resp = st.session_state.llm.invoke(prompt)
                answer_text = getattr(resp, "content", str(resp))
            except Exception as e:
                answer_text = f"LLM error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer_text})
        with st.chat_message("assistant"):
            st.markdown(answer_text)

        # Do not show retrieved documents or images; only show the answer above

# Optional PDF viewer
if 'show_pdf' in locals() and show_pdf:
    st.divider()
    st.subheader("Source PDF")
    render_pdf(pdf_path)


