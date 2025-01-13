import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import pinecone

class LangchainPipeline:
    def __init__(self, directory: str, pinecone_api_key: str, index_name: str, openai_api_key: str):
        self.directory = directory
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.openai_api_key = openai_api_key

        pinecone.init(api_key=self.pinecone_api_key)
        self.documents: Optional[List[Document]] = None
        self.chunked_documents: Optional[List[Document]] = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.index: Optional[LangchainPinecone] = None
        self.llm: Optional[OpenAI] = None
        self.chain = None

    def read_documents(self) -> List[Document]:
        file_loader = PyPDFDirectoryLoader(self.directory)
        self.documents = file_loader.load()
        return self.documents

    def chunk_documents(self, chunk_size: int = 800, chunk_overlap: int = 50) -> List[Document]:
        if not self.documents:
            raise ValueError("No documents loaded. Call read_documents() first.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunked_documents = text_splitter.split_documents(self.documents)
        return self.chunked_documents

    def initialize_embeddings(self) -> OpenAIEmbeddings:
        self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        return self.embeddings

    def create_pinecone_index(self) -> LangchainPinecone:
        if not self.chunked_documents:
            raise ValueError("No chunked documents. Call chunk_documents() first.")
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Call initialize_embeddings() first.")
        self.index = LangchainPinecone.from_documents(
            self.chunked_documents,
            self.embeddings,
            index_name=self.index_name
        )
        return self.index

    def retrieve_query(self, query: str, k: int = 2) -> List[Document]:
        if not self.index:
            raise ValueError("Pinecone index not initialized. Call create_pinecone_index() first.")
        return self.index.similarity_search(query, k=k)

    def initialize_llm_chain(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.5):
        self.llm = OpenAI(model_name=model_name, temperature=temperature)
        self.chain = load_qa_chain(self.llm, chain_type="stuff")
        return self.chain

    def retrieve_answers(self, query: str) -> str:
        if not self.chain:
            raise ValueError("LLM chain not initialized. Call initialize_llm_chain() first.")
        doc_search = self.retrieve_query(query)
        return self.chain.run(input_documents=doc_search, question=query)
