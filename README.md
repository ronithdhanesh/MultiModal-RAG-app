# Multimodal RAG with Streamlit

### Project Overview

This is a multimodal Retrieval-Augmented Generation (RAG) application built with Python and Streamlit. It allows users to chat with a PDF document by answering questions based on both the text and images contained within it. The application extracts content from the PDF, creates a searchable vector database, and uses a large language model (LLM) to generate accurate, context-aware responses.

The app is deployed and live on Streamlit Cloud.

### Key Features

* **Multimodal Retrieval:** Retrieves relevant information from both text and images within the PDF document.
* **Intuitive Chat Interface:** A user-friendly Streamlit chat interface for interacting with the document.
* **Fast & Efficient:** Uses a pre-built vector database for near-instant retrieval, avoiding costly and slow indexing on every query.
* **PDF Viewer:** Includes an optional PDF viewer in the sidebar for easy reference.

### How It Works

1.  **Data Ingestion:** The application processes the PDF, extracting text and images.
2.  **Vector Database:** The extracted text and images are converted into numerical embeddings and stored in a vector database (`outputs/vectorstore`).
3.  **Retrieval:** When a user asks a question, the app searches the vector database to find the most relevant text chunks and images.
4.  **Generation:** The retrieved context (text and image references) is provided to a Google Gemini model, which uses it to formulate a precise answer.

### Local Installation & Usage

To run this project on your local machine, follow these steps:

#### Prerequisites

* Python 3.8+
* `git`

#### Steps

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your_username]/[your_repo_name].git
    cd [your_repo_name]
    ```

2.  **Set Up a Virtual Environment:**
    ```bash
    python -m venv venv
    # On macOS/Linux
    source venv/bin/activate
    # On Windows
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Keys:**
    Create a new directory named `.streamlit` in your project's root. Inside that directory, create a file named `secrets.toml` and add your Google API key:
    
    ```toml
    # .streamlit/secrets.toml
    GOOGLE_API_KEY = "your_api_key_here"
    ```

5.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
    The app should open in your default web browser.

### Deployment

This app is designed to be deployed on **Streamlit Cloud**. The `requirements.txt` and pre-built `outputs/` folder (containing the vector database) ensure a fast and seamless deployment process.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

### Acknowledgments

### Acknowledgments

* Built with [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), and [PyMuPDF](https://pypi.org/project/PyMuPDF/).
* **Multimodal embeddings powered by CLIP.**
* Powered by the Google Gemini model.
