from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts=[
  "Hello, this is a test of MistralAI's embedding model.",
  "This is another example of text to embed.",
  "MistralAI provides powerful embedding capabilities."
]

embeddings_list = embedding.embed_documents(texts)
print(f"Generated {len(embeddings_list)} embeddings.")

print(f"Dimension of first embedding: {len(embeddings_list)}")