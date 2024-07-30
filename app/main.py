from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import tokenizer, model, embeddings_model
import pinecone

app = FastAPI()

class Query(BaseModel):
    user_query: str

index_name = "infiheal"
index = pinecone.Index(host="PINECONE-HOST-NAME", api_key="PINECONE-API-KEY")

def get_response(user_query):
    query_embedding = embeddings_model.encode(user_query).tolist()

    # Perform a similarity search in the index
    matching_ids = index.query(vector=query_embedding, top_k=3)
    content_list = []
    source_list = []
    for id in matching_ids['matches']:
        result = index.fetch(ids=[id['id']])
        content = result['vectors'][str(id['id'])]['metadata']['content']
        source = result['vectors'][str(id['id'])]['metadata']['source']
        content_list.append(content)
        source_list.append(source)

    prompt = (
        """
        Use the following pieces of information to answer the user's question.

        Context: {context}
        Question: {question}
        Helpful answer:
        """
    ).format(context=content_list, question=user_query)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(input_ids, max_new_tokens=600)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the part after the prompt "Helpful answer:"
    answer_start = generated_text.find("Helpful answer:") + len("Helpful answer:")
    answer = generated_text[answer_start:].strip()

    # Use a set to keep track of unique sources
    unique_sources = set()
    unique_metadatas = []

    # Iterate through the metadata list and add only unique sources to the final list
    for metadata in source_list:
        if metadata not in unique_sources:
            unique_sources.add(metadata)
            unique_metadatas.append(metadata)

    return answer, unique_metadatas

def get_category(user_query):

    prompt = (
        """
        User's Question: {question}
        Categories: General, Causes, Symptoms, Treatment, Prevention, Support
        Suggest the most suitable category from above specified Categories
        according to user's question.

        Example:
        Question: What is mental health?
        Category: General
        Question: How to handle depression?
        Category: Treatment
        Question: What are the symptoms of depression?
        Category: Symptoms

        Category for the question is?
        """
    ).format(question=user_query)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=150)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("generated_text:", generated_text)

    # Extract the category from the generated text
    category_prompt = "Category for the question is?"
    category_start = generated_text.find(category_prompt) + len(category_prompt)
    if category_start == -1:
        raise ValueError("Category prompt not found in the generated text")

    # Extract the text after the category prompt
    category_text = generated_text[category_start:].strip()

    # Clean up to only get the category
    category = category_text.split('\n')[0].strip()  # Assuming the category is on the first line

    return category

@app.post("/query")
def query(user_query: Query):
    try:
        response, sources = get_response(user_query.user_query)
        return {"response": response, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classification")
def classify(user_query: Query):
    try:
        category = get_category(user_query.user_query)
        return {"category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
