from config import embeddings_model
import numpy as np
import requests
from newspaper import Article # https://github.com/codelucas/newspaper
import time
import nltk
from nltk.tokenize import word_tokenize

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

# We can add more URLs as required
article_urls = [
    "https://www.cdc.gov/mentalhealth/learn/index.htm",
    "https://www.acko.com/health-insurance/important-facts-about-mental-health/",
    "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response",
    "https://www.medicalnewstoday.com/articles/154543#definition"
]

session = requests.Session()
pages_content = [] # where we save the scraped articles

for url in article_urls:
    try:
        time.sleep(2) # sleep two seconds for gentle scraping
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            article = Article(url)
            article.download() # download HTML of webpage
            article.parse() # parse HTML to extract the article text
            pages_content.append({ "url": url, "text": article.text })
        else:
            print(f"Failed to fetch article at {url}")
    except Exception as e:
        print(f"Error occurred while fetching article at {url}: {e}")

#If an error occurs while fetching an article, we catch the exception and print
#an error message. This ensures that even if one article fails to download,
#the rest of the articles can still be processed.

# Download the punkt tokenizer model
nltk.download('punkt')

def split_text_by_words(text, chunk_size, chunk_overlap):
    words = word_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if current_length + word_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[int(chunk_overlap/2):]
            current_length = sum(len(w) + 1 for w in current_chunk)
        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

chunk_size = 1000
chunk_overlap = 250

all_texts = []
all_metadatas = []

for d in pages_content:
    chunks = split_text_by_words(d["text"], chunk_size, chunk_overlap)
    for chunk in chunks:
        all_texts.append(chunk)
        all_metadatas.append({"source": d["url"]})

# Function to generate embeddings for text chunks
def generate_embeddings(chunks, embeddings_model):
    embeddings = []
    for chunk in chunks:
        chunk_embedding = embeddings_model.encode(chunk)  # Changed to use Sentence Transformer
        embeddings.append(chunk_embedding)
    return np.array(embeddings)

# Generate embeddings for text chunks
embeddings = generate_embeddings(all_texts, embeddings_model)

index_name = 'INDEX_NAME'

# import pinecone
import numpy as np
import os
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone client with your API key and index name
pc = Pinecone(
    api_key= "PINECONE-API-KEY"
)

# Now do stuff
if 'INDEX_NAME' not in pc.list_indexes().names():
    pc.create_index(
        name='INDEX_NAME',
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Generate sequential numeric IDs
ids = list(range(1, len(embeddings) + 1))

embeddings_list = embeddings.tolist()

#Upsert embeddings into the index
for emb, id, chunk,source in zip(embeddings_list, ids, all_texts, all_metadatas):
  index.upsert(vectors=[{"id": str(id), "values": emb, "metadata": {"source":str(source),"content" : chunk} }])