import os
import re
import shutil
import logging

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def extract_file_metadata(source):
    filename = os.path.basename(source)
    filename_lower = filename.lower()

    company = "Unknown"

    if "bmw" in filename_lower:
        company = "BMW"
    elif "tesla" in filename_lower:
        company = "Tesla"
    elif "ford" in filename_lower:
        company = "Ford"

    year_match = re.search(r"(20\d{2})", filename)

    year = None

    if year_match:
        year = int(year_match.group(1))

    return company, year, filename

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    logging.info(
        f"Loaded {len(documents)} pages from PDFs."
    )

    for doc in documents:

        source = doc.metadata.get("source", "")

        company, year, filename = extract_file_metadata(
            source
        )

        doc.metadata["company"] = company
        doc.metadata["year"] = year
        doc.metadata["filename"] = filename

        page = doc.metadata.get("page")

        if page is not None:
            doc.metadata["page_number"] = int(page) + 1

    return documents

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    logging.info(
        f"Created {len(chunks)} chunks."
    )

    return chunks

def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

def build_vectorstore():

    documents = load_pdf_files(DATA_PATH)

    if not documents:
        raise ValueError(
            "No PDF files found in the data folder."
        )

    chunks = create_chunks(documents)

    embedding_model = get_embedding_model()

    if os.path.exists(DB_FAISS_PATH):
        logging.info(
            "Removing existing FAISS database..."
        )
        shutil.rmtree(DB_FAISS_PATH)

    logging.info(
        "Creating embeddings and FAISS database..."
    )

    db = FAISS.from_documents(
        chunks,
        embedding_model
    )

    os.makedirs(
        os.path.dirname(DB_FAISS_PATH),
        exist_ok=True
    )

    db.save_local(DB_FAISS_PATH)

    logging.info(
        f"FAISS database saved to: {DB_FAISS_PATH}"
    )


if __name__ == "__main__":

    build_vectorstore()