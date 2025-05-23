---
layout: post
title: CRM with AI providing the magic.
image: "/posts/AI_Assistant_crm.png"
tags: [ML,CRM, LLM]
---


Okay, this is a fantastic piece of code! It's building a **Streamlit web application** that acts as an intelligent **Insurance Licensing & Product Training Assistant.** Essentially, it's designed to help someone study for insurance exams by letting them "chat" with their PDF training materials and even generate quizzes from them.

This is a prime example of a **Retrieval Augmented Generation (RAG)** system. Let's break down this analysis into what's happening, why it's clever, and how it all comes together for the user.

**The Big Picture: Your Own AI Insurance Tutor**

Imagine you have a pile of PDF documents – core licensing guides, product training manuals, etc. Instead of manually searching through them or trying to memorize everything, this application does the following:

1. **Reads & Understands Your PDFs:** It ingests all the text from these documents.
2. **Creates a Smart Index:** It builds a special, searchable "brain" or index of this information.
3. **Lets You Ask Questions:** You can type in a question (e.g., "What are the core principles of utmost good faith?") and the system will find the relevant information in your PDFs and use an AI to give you an answer.
4. **Generates Quizzes:** You can ask it to create a quiz on a specific topic (or general topics) based _only_ on the content of your documents, helping you test your knowledge.

**Let's Dive into the "Magic" – Key Components & Concepts:**

**Phase 1: Setting the Stage (Imports and Configuration)**

Python

```
import streamlit as st # For building the web app interface
import os, glob, fitz # For handling files, folders, and reading PDFs (PyMuPDF)
from dotenv import load_dotenv # For managing secret keys (like an OpenAI API key)

# Langchain components (The AI Brains & Connectors)
from langchain_community.vectorstores import Chroma # Our "smart filing cabinet" for documents
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # Tools to understand language and chat with OpenAI's AI
from langchain.text_splitter import RecursiveCharacterTextSplitter # For breaking down big documents
from langchain.chains import RetrievalQA # For the Question & Answering system
from langchain.prompts import PromptTemplate # For giving clear instructions to the AI
from langchain.docstore.document import Document # A standard way to handle pieces of text

# Load environment variables
load_dotenv()

# --- Configuration ---
CORE_PDF_DIR = "core_licensing_pdfs"
PRODUCT_PDF_DIR = "product_training_pdfs"
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "insurance_docs"
```

- **What's happening?** We're importing all the necessary libraries (toolkits) and setting up some basic configurations.
    - `streamlit`: This is what will create the interactive webpage you see and use.
    - `fitz (PyMuPDF)`: A powerful tool for extracting text directly from PDF files, page by page.
    - `Langchain`: This is a popular framework that makes it much easier to build applications powered by Large Language Models (LLMs) like OpenAI's GPT.
        - `Chroma`: This will be our **Vector Store**. Think of it as a super-efficient, searchable database specifically designed to store and retrieve information based on _meaning_ and _context_, not just keywords.
        - `OpenAIEmbeddings`: This is the magic that turns text into **embeddings** (numerical representations or "vectors"). Essentially, it converts words and sentences into a list of numbers that captures their meaning. Similar concepts will have similar numbers.
        - `ChatOpenAI`: This is how our application will "talk" to an OpenAI model (like GPT-3.5-turbo) to generate answers and quizzes.
        - `RecursiveCharacterTextSplitter`: PDFs can be long! LLMs have a limit to how much text they can "read" at once (their "context window"). This tool cleverly breaks down the PDF text into smaller, overlapping chunks.
        - `RetrievalQA`: This is a pre-built Langchain "chain" that orchestrates the process of: 1) taking your question, 2) retrieving relevant chunks from the Vector Store, and 3) sending those chunks plus your question to the LLM to get an answer.
        - `PromptTemplate`: When we ask the AI to do something complex (like generate a quiz), we need to give it very clear instructions. This tool helps us create structured "prompts."
    - `load_dotenv()`: This loads any secret keys (like your OpenAI API key) from a special `.env` file, so you don't have to hardcode them.
    - **Configuration variables:** These define where your PDFs are stored (`CORE_PDF_DIR`, `PRODUCT_PDF_DIR`), where the "smart filing cabinet" (ChromaDB) will be saved (`CHROMA_PERSIST_DIR`), and what it will be called (`CHROMA_COLLECTION_NAME`).

**Phase 2: Reading and Understanding the PDFs (The Librarian's Work)**

Python

```
def load_pdfs_from_directory_pymupdf(directory_path):
    # ... (code to find PDF files) ...
    # ... (loop through each PDF) ...
        doc_pymupdf = fitz.open(pdf_path) # Open the PDF
        text = ""
        for page_num in range(len(doc_pymupdf)): # For each page
            page = doc_pymupdf.load_page(page_num)
            page_text = page.get_text("text") # Extract plain text
            if page_text:
                text += page_text + "\n\n" # Collect all text from the PDF
        # Create a Langchain Document object
        doc = Document(page_content=text.strip(), metadata={"source": os.path.basename(pdf_path)})
        all_docs.append(doc)
    # ... (progress bar updates) ...
    return all_docs
```

- **What's happening?** This function is like a diligent librarian.
    
    1. It looks in a specified folder (`directory_path`) for all PDF files.
    2. For each PDF, it opens it using `fitz (PyMuPDF)`.
    3. It goes through every single page and extracts all the visible text.
    4. It combines all the text from one PDF into a single block.
    5. Crucially, it then wraps this text into a Langchain `Document` object. This object not only holds the `page_content` (the text) but also `metadata` – in this case, the `source` (the filename of the PDF), which is super helpful later for citing sources! <!-- end list -->
    
    - **Why is this brilliant?** It systematically converts your static PDF library into a collection of digital text that the AI can work with, and it keeps track of where each piece of text came from. The progress bar (`st.progress`) is a nice touch for the user interface, letting them know what's happening with large collections of PDFs.

**Phase 3: Building the "Brain" – The Vector Store & Retriever (The Super-Smart Filing System)**

Python

```
@st.cache_resource(ttl="1h") # Cache this expensive operation!
def initialize_vector_store_and_retriever():
    # 1. Initialize LLM and Embeddings
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    embeddings = OpenAIEmbeddings()

    # 2. Check if Chroma DB (Vector Store) exists or needs rebuilding
    # ... (logic to check for existing DB or if user forced a rebuild) ...

    # 3. If DB needs to be built:
    if vector_store is None:
        # 3a. Load PDFs
        core_docs = load_pdfs_from_directory_pymupdf(CORE_PDF_DIR)
        product_docs = load_pdfs_from_directory_pymupdf(PRODUCT_PDF_DIR)
        all_documents = core_docs + product_docs
        # ... (error handling if no documents) ...

        # 3b. Split Documents into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_documents)
        # ... (error handling if no chunks) ...

        # 3c. Create and Persist ChromaDB
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings, # Use OpenAI to "understand" and vectorize chunks
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )
        vector_store.persist() # Save it to disk

    # 4. Create a Retriever
    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever, llm
    # ... (error handling) ...
```

- **What's happening?** This is the most complex and crucial part – building the core RAG engine.
    
    1. **Initialize AI Components:** It sets up the `ChatOpenAI` model (the "brain" that will answer questions and generate quizzes, `temperature=0.2` makes its responses more focused and less random) and the `OpenAIEmbeddings` model (the "understanding" part that turns text into numbers).
    2. **Vector Store Management:**
        - It cleverly checks if a `ChromaDB` (our vector store / smart filing cabinet) already exists in the `CHROMA_PERSIST_DIR`. If you've run the app before and processed your PDFs, it tries to load the existing one to save time.
        - It also allows the user to `force_rebuild_db` (e.g., if new PDFs are added).
    3. **Building a New Vector Store (if needed):**
        - **Load PDFs:** It calls our `load_pdfs_from_directory_pymupdf` function to get all the text from both "core" and "product" PDF directories.
        - **Split Documents:** This is key! The `RecursiveCharacterTextSplitter` takes all that text and breaks it into smaller, manageable chunks (e.g., 1500 characters each, with a 200-character overlap to ensure context isn't lost between chunks). This is vital because LLMs have limits on how much text they can process at once.
        - **Create & Persist ChromaDB:** This is where the magic happens!
            - `Chroma.from_documents(...)`: For each `split_doc` (chunk of text), it uses the `OpenAIEmbeddings` to convert that chunk into a numerical vector (a list of numbers representing its meaning).
            - It then stores these vectors (along with the original text chunks) in the `Chroma` vector store.
            - `vector_store.persist()`: It saves this newly built vector store to your computer disk in the `CHROMA_PERSIST_DIR` so it can be quickly loaded next time.
    4. **Create a Retriever:** Once the `vector_store` is ready (either loaded or newly built), it creates a `retriever`.
        - The `retriever`'s job is simple but powerful: when you give it a query (like a user's question), it efficiently searches the `vector_store` to find the text chunks whose meanings (their vectors) are most similar to the query's meaning.
        - `search_kwargs={"k": 5}` means it will retrieve the top 5 most relevant chunks.
    
    - **Why is this brilliant?**
        - `@st.cache_resource`: This Streamlit decorator is a lifesaver! Building the vector store (especially embedding lots of text) can be slow and computationally expensive (it costs money if using OpenAI embeddings). This caches the `retriever` and `llm` objects, so they are only re-created if necessary (e.g., if you force a rebuild or after the cache `ttl` – time to live – expires, here 1 hour). This makes the app much faster on subsequent runs or interactions.
        - **Persistence:** Saving the ChromaDB means you don't re-process all PDFs every time you start the app.
        - **Chunking Strategy:** Smart splitting helps ensure relevant context is fed to the LLM without overwhelming it.

**Phase 4: Setting Up the Question-Answering and Quiz Systems**

Python

```
def get_qa_chain(_llm, _retriever):
    # ...
    return RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff", # "Stuff" all retrieved docs into the prompt
        retriever=_retriever,
        return_source_documents=True # So we can show where the answer came from!
    )

def generate_quiz_llm(_llm, _retriever, topic=None, num_questions=3, question_type="multiple_choice"):
    # 1. Retrieve context based on the topic (or general info if no topic)
    context_docs = _retriever.get_relevant_documents(search_query)
    # ... (error handling) ...
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    # ... (truncate context_text if too long for LLM) ...

    # 2. Define a detailed Prompt Template for the LLM
    quiz_prompt_template_str = """Based SOLELY on the following context, generate a ... quiz ...
    Ensure questions are directly and unambiguously answerable from the provided context ONLY. Do not use outside knowledge.
    Context:
    {context}
    Quiz:
    """
    quiz_prompt = PromptTemplate(...)
    formatted_prompt = quiz_prompt.format(...)

    # 3. Ask the LLM to generate the quiz
    response = _llm.invoke(formatted_prompt)
    return response.content
```

- **What's happening?**
    - `get_qa_chain`: This function sets up the Langchain `RetrievalQA` chain.
        - When a user asks a question, this chain will:
            1. Use the `_retriever` to find the top `k` relevant text chunks from your PDFs.
            2. "Stuff" these chunks (the `chain_type="stuff"` part) along with the user's question into a prompt for the `_llm`.
            3. The `_llm` then generates an answer based _only_ on this provided context.
            4. `return_source_documents=True` is fantastic because it means we can show the user _which_ parts of their documents were used to generate the answer, building trust and allowing verification.
    - `generate_quiz_llm`: This function is custom-built to generate quizzes.
        1. **Retrieve Context:** It first uses the `_retriever` to find relevant document chunks based on an optional `topic` provided by the user. If no topic, it uses a general search query.
        2. **Careful Prompting:** It uses a `PromptTemplate` to give very specific instructions to the `_llm`. This prompt emphasizes that the quiz must be based **SOLELY** on the provided `context` (the retrieved document chunks) and not on the LLM's general knowledge. This is the core of making RAG work reliably for factual tasks. It also specifies the number of questions and question type.
        3. **LLM Invocation:** It sends this carefully crafted prompt (with the retrieved context embedded) to the `_llm` and gets back the generated quiz.
    - **Why is this brilliant?**
        - **Grounded Responses:** Both Q&amp;A and quiz generation are grounded in _your documents_. The LLM isn't just making things up; it's synthesizing information from the provided context. This is crucial for accuracy in a domain like insurance licensing.
        - **Clear Instructions to AI:** The `PromptTemplate` for the quiz is a great example of "prompt engineering" – telling the AI exactly what you want and how you want it, especially the constraint to use _only_ the provided context.

**Phase 5: The User Interface (Streamlit Magic)**

Python

```
# --- Streamlit App UI ---
st.set_page_config(page_title="Insurance RAG Assistant", layout="wide")
st.title("📚 Insurance Licensing & Product Training Assistant")
# ... (markdown welcome message) ...

# --- Initialize directories and dummy files if they don't exist ---
# ... (code to create dummy PDFs for first-time users using reportlab) ...
# This is a thoughtful addition for users to test the app without their own PDFs initially.

# --- Sidebar for controls ---
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Force Rebuild Vector DB", ...):
# ... (logic to clear cache and set rebuild flag) ...
# ... (sidebar info about PDF processing, irrelevant info, OCR) ...

# --- Main Application Logic ---
retriever, llm = initialize_vector_store_and_retriever() # This is cached!

if retriever and llm: # Only proceed if initialization was successful
    qa_chain = get_qa_chain(llm, retriever)

    # --- Q&A Section ---
    st.header("💬 Ask a Question")
    user_question = st.text_input(...)
    if user_question:
        # ... (invoke qa_chain, display answer and sources) ...

    # --- Quiz Generation Section ---
    st.header("📝 Generate a Quiz")
    # ... (UI elements for topic, num_questions, quiz_type) ...
    if st.button("✨ Generate Quiz", ...):
        # ... (call generate_quiz_llm, store result in session_state) ...
    if 'quiz_result' in st.session_state and st.session_state.quiz_result:
        st.subheader("Generated Quiz:")
        st.markdown(st.session_state.quiz_result) # Display quiz
else:
    st.error("Models or Vector Store could not be initialized...")
```

- **What's happening?** This is all the Streamlit code that creates the interactive web page.
    - `st.set_page_config`, `st.title`, `st.markdown`: Basic page setup and text display.
    - **Dummy PDF Creation:** A very user-friendly feature! If the app starts and finds no PDFs in your folders, it tries to create a couple of simple dummy PDFs using the `reportlab` library. This allows a new user to immediately see the app working without having to find their own documents first. It then prompts a `st.rerun()` to ensure the app picks up these new files.
    - **Sidebar Controls:**
        - A button to "Force Rebuild Vector DB" gives the user control to re-process documents if they've changed. This cleverly clears the cache for `initialize_vector_store_and_retriever` and uses `st.session_state` (Streamlit's way to remember things across interactions) to trigger the rebuild.
        - Informative text about PDF processing, handling irrelevant info (a known challenge in RAG), and OCR (for image-based PDFs).
    - **Main Logic:**
        - It first calls `initialize_vector_store_and_retriever()` to get the core RAG components. Thanks to caching, this will be fast unless a rebuild is forced or the cache expires.
        - **Q&amp;A Section:** Provides a text input box for the user. When they type a question and press Enter (or if a button was used), it invokes the `qa_chain`, shows a "spinner" while waiting, and then displays the LLM's answer and the source documents in an expandable section.
        - **Quiz Section:** Provides inputs for quiz topic, number of questions, and question type. When the "Generate Quiz" button is clicked, it calls `generate_quiz_llm`. The generated quiz is stored in `st.session_state.quiz_result` so it persists across Streamlit reruns (which happen with almost every widget interaction) and is then displayed.
    - **Why is this brilliant?**
        - **User-Friendly Interface:** Streamlit makes it easy to create a clean and interactive UI without needing to be a web development expert.
        - **State Management:** `st.session_state` is used effectively to manage things like whether a rebuild is needed or to store the quiz result.
        - **Error Handling and User Feedback:** The code includes `st.spinner` for long operations, `st.error` for issues, `st.warning`, and `st.success` messages, which greatly improve the user experience.
        - **Thoughtful Onboarding:** The dummy PDF creation and sidebar information show consideration for the user's journey.

**In Summary: A Powerful, User-Friendly Study Tool**

This code is a sophisticated example of building a practical RAG application. It intelligently combines:

1. **Efficient Document Processing:** Extracting text from potentially many PDFs.
2. **Semantic Search Power:** Using embeddings and a vector store (ChromaDB) to find relevant information based on meaning.
3. **LLM Intelligence:** Leveraging an OpenAI model (GPT-3.5-turbo) for natural language understanding, question answering, and quiz generation.
4. **Grounding & Control:** Ensuring the LLM's responses are based on the provided documents through careful prompting and the RAG architecture.
5. **User-Friendly Web Interface:** Using Streamlit to make the tool accessible and interactive.
6. **Performance Considerations:** Caching resource-intensive operations and providing a persistent vector store.

It's a "brilliant" solution because it takes complex AI concepts and packages them into a genuinely useful tool that can significantly aid someone in studying dense technical material by making that material interactive and queryable in natural language.
