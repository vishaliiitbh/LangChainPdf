import os
from tkinter import Tk, Label, Entry, Button, filedialog, Text, Scrollbar, END
from llm import LangchainPipeline
from dotenv import load_dotenv

class LangchainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LangChain Pipeline App")
        self.pipeline = None

        # Directory selection
        Label(root, text="PDF Directory:").grid(row=0, column=0, padx=10, pady=10)
        self.directory_entry = Entry(root, width=50)
        self.directory_entry.grid(row=0, column=1, padx=10, pady=10)
        Button(root, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=10, pady=10)

        # Query input
        Label(root, text="Query:").grid(row=1, column=0, padx=10, pady=10)
        self.query_entry = Entry(root, width=50)
        self.query_entry.grid(row=1, column=1, padx=10, pady=10)

        # Submit button
        Button(root, text="Get Answer", command=self.get_answer).grid(row=2, column=0, columnspan=3, pady=10)

        # Response display
        Label(root, text="Response:").grid(row=3, column=0, padx=10, pady=10)
        self.response_text = Text(root, wrap="word", height=10, width=60)
        self.response_text.grid(row=3, column=1, columnspan=2, padx=10, pady=10)
        scrollbar = Scrollbar(root, command=self.response_text.yview)
        scrollbar.grid(row=3, column=3, sticky="nsew")
        self.response_text.config(yscrollcommand=scrollbar.set)

    def browse_directory(self):
        directory = filedialog.askdirectory()
        self.directory_entry.delete(0, END)
        self.directory_entry.insert(0, directory)

    def get_answer(self):
        directory = self.directory_entry.get()
        query = self.query_entry.get()

        if not directory or not query:
            self.response_text.insert(END, "Please provide a valid directory and query.\n")
            return

        try:
            # Load environment variables
            load_dotenv()
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            openai_api_key = os.getenv("OPENAI_API_KEY")

            # Initialize the pipeline
            self.pipeline = LangchainPipeline(
                directory=directory,
                pinecone_api_key=pinecone_api_key,
                index_name="langchainvector",
                openai_api_key=openai_api_key
            )

            # Process
            self.pipeline.read_documents()
            self.pipeline.chunk_documents()
            self.pipeline.initialize_embeddings()
            self.pipeline.create_pinecone_index()
            self.pipeline.initialize_llm_chain()

            # Get the answer
            answer = self.pipeline.retrieve_answers(query)
            self.response_text.delete(1.0, END)
            self.response_text.insert(END, answer)

        except Exception as e:
            self.response_text.delete(1.0, END)
            self.response_text.insert(END, f"Error: {str(e)}\n")


if __name__ == "__main__":
    root = Tk()
    app = LangchainApp(root)
    root.mainloop()
