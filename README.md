LLMDEV provides an end-to-end pipeline for processing medical Instructions for Use (IFU) PDFs, generating embeddings, storing vectors, and enabling Retrieval-Augmented Generation (RAG) using Azure OpenAI or OpenAI models.

Pipeline Overview

Upload IFU PDF → Azure Blob Storage

Download & extract PDF text

Chunk text using IFUChunker

tokenisation

language detection

overlapping sliding windows

Embed chunks (Azure/OpenAI embeddings)

Store vectors in ChromaDB

Query with semantic similarity

Return grounded answers via LLM chat model

Key Components

AzureBlobLoader – upload, download, extract PDF text

IFUChunker – split IFU text into language-aware chunks

IFUEmbedder – batch embedding using Azure/OpenAI

ChromaIFUVectorStore – store & search embeddings

IFUChatModel – build RAG responses

Config – centralised environment variable management

Tests – pipeline, health checks, chunking, embeddings

LLMDEV/
├── azure/
├── chunking/
├── embedding/
├── vectorstore/
├── chat/
├── config/
├── tests/
└── data/

loader = AzureBlobLoader(cfg)
blob = loader.upload_pdf("ifu.pdf")
text = loader.download_and_extract(blob)

chunks = IFUChunker(tokenizer, lang_detector).chunk_text(text)
embeddings = IFUEmbedder(cfg).embed_chunks(chunks)

store = ChromaIFUVectorStore(cfg)
store.add_chunks(embeddings)

results = store.query_text("How do I maintain the ankle?")

pip install -r requirements.txt
export AZURE_STORAGE_ACCOUNT="..."
export OPENAI_API_KEY="..."
export CHROMA_TENANT="..."
