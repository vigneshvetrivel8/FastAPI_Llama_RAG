# Instructions to Run the Code
### Step 1: Create and Activate a Virtual Environment (Optional)
virtualenv venv  
source venv/bin/activate  
### Step 2: Install Dependencies
pip install -r requirements.txt  
### Step 3: Create Vector Embeddings and Upsert to Pinecone (Optional and One-Time)  
python app/initialize.py
### Step 4: Run the FastAPI Server  
uvicorn app.main:app --reload  

# Project Structure
### app/config.py
This file is responsible for loading the model, tokenizer, and embeddings model for the LLM used in the project.

### app/initialize.py
This script handles the upserting of data into the vector database.  
It:  
Extracts content from the provided URLs.  
Chunks the content into manageable pieces.  
Embeds the chunks and upserts them into the vector database.  
### app/main.py
This is the main FastAPI application file, containing endpoints for generating responses and categorizing queries.

/rag Endpoint:  
Generates a response to the user's query and provides related sources or articles for further reading.  
/classification Endpoint:  
Categorizes the user's query into predefined categories such as General, Causes, Symptoms, Treatment, Prevention, and Support.  