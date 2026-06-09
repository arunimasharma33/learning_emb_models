from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()

# Initialize MistralAI embeddings client
embeddings = MistralAIEmbeddings(
    model="mistral-embed",          # Mistral's embedding model
    
)

# Example: embed a text
text = "Hello, this is a test of MistralAI's embedding model."
texts=[
  "Hello, this is a test of MistralAI's embedding model.",
  "This is another example of text to embed.",
  "MistralAI provides powerful embedding capabilities."
]
sample = embeddings.embed_query("Test")
print(len(sample))
embeddings_list = embeddings.embed_documents(texts)

print(f"Embeddings for multiple texts: {embeddings_list}")