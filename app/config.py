from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"

# Load the models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')