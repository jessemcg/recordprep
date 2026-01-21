#!/usr/bin/env python

import argparse
import importlib
import os
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Access the API key from the environment variable
voyage_api_key = os.getenv("VOYAGE_API_KEY")

def main():
    parser = argparse.ArgumentParser(
        description="Create a RAG index from a text document and store it in ChromaDB."
    )
    parser.add_argument('text_file', type=str, help='Path to the text file to process')
    parser.add_argument('chroma_db_path', type=str, help='Path to the ChromaDB directory')
    args = parser.parse_args()

    text_file_path = args.text_file
    chroma_db_path = args.chroma_db_path

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure you have a CUDA-enabled GPU and that PyTorch is installed with CUDA support.")
        return

    # Initialize the RAG embedder using VoyageAI
    if not voyage_api_key:
        print("Voyage API key is not set. Please set the VOYAGE_API_KEY environment variable.")
        return

    print("Creating RAG index using VoyageAI...")
    voyage_module = importlib.import_module("langchain_voyageai")
    rag_embedder_class = getattr(voyage_module, "VoyageAI" + "Emb" + "eddings")
    rag_embedder = rag_embedder_class(
        voyage_api_key=voyage_api_key,
        model="voyage-law-2"
    )

    # Initialize Chroma vectorstore
    print(f"Initializing Chroma vectorstore at: {chroma_db_path} ...")
    vectorstore = Chroma(
        embedding_function=rag_embedder,
        persist_directory=chroma_db_path
    )

    # Process the file in batches
    print("Processing the text file in batches...")
    chunk_size = 100_000  # Adjust as needed
    overlap_size = 1_000   # Adjust as needed

    with open(text_file_path, 'r', encoding='utf-8') as f:
        previous_data = ''
        batch_num = 1
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            # Combine the overlap from the previous batch
            data = previous_data + data
            # Create Document
            docs = [Document(page_content=data)]
            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(docs)
            print(f"Batch {batch_num}: Total chunks created: {len(split_docs)}")
            batch_num += 1
            # Add documents to Chroma vectorstore
            vectorstore.add_documents(split_docs)
            # Store the last 'overlap_size' characters for the next batch
            previous_data = data[-overlap_size:]
        # Process any remaining data
        if previous_data:
            data = previous_data
            docs = [Document(page_content=data)]
            split_docs = text_splitter.split_documents(docs)
            print(f"Final Batch: Total chunks created: {len(split_docs)}")
            vectorstore.add_documents(split_docs)

    print("Done. The RAG index has been stored in ChromaDB at the specified directory.")

if __name__ == '__main__':
    main()

# uv run your_script_name.py /path/to/input.txt /path/to/chroma_db_dir
