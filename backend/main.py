from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import initialize_agent, AgentType, AgentExecutor
import os
import docx
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

app = FastAPI()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\MGSLIC\Downloads\precise-sight-471404-b4-62655cfb649a.json"

# Store retriever tools globally
retriever_tools = {}

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.7,
    google_api_key="AIzaSyApls8O55HoO8-YnUBbpmDdbn5xGGjpzY4"
)

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Global vectorstore map
vectorstores = {}

# Pydantic model for chat request
class ChatRequest(BaseModel):
    message: str


# -------- Unified Upload Endpoint -------- #
@app.post("/upload_source/")
async def upload_source(file: UploadFile = None, url: str = Form(None)):
    try:
        content = ""
        source_name = ""

        if file:  # File upload
            ext = os.path.splitext(file.filename)[-1].lower()
            source_name = file.filename

            if ext == ".txt":
                content = (await file.read()).decode("utf-8")
            elif ext == ".pdf":
                pdf_bytes = await file.read()
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page in pdf_doc:
                    content += page.get_text()
            elif ext == ".docx":
                temp_path = f"temp_{file.filename}"
                with open(temp_path, "wb") as f:
                    f.write(await file.read())
                doc = docx.Document(temp_path)
                content = "\n".join([p.text for p in doc.paragraphs])
                os.remove(temp_path)
            else:
                return JSONResponse({"error": "Unsupported file type"}, status_code=400)

        elif url:  # URL input
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return JSONResponse({"error": "Failed to fetch URL"}, status_code=400)

            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator="\n")
            content = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            source_name = url.replace("https://", "").replace("http://", "")

        else:
            return JSONResponse({"error": "No file or URL provided"}, status_code=400)

        # Chunk and embed
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50).split_text(content)
        vectorstore = FAISS.from_texts(chunks, embedding)

        # Save vectorstore
        vectorstores[source_name] = vectorstore

        # Create retriever tool
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        retriever_tools[source_name] = {
            "retriever": retriever,
            "tool": create_retriever_tool(
            retriever,
            name=f"{source_name}_retriever",
            description=f"Search knowledge inside {source_name}."
        )
    }

        return {
            "source": source_name,
            "stored_chunks": len(chunks),
            "content_preview": content[:500]
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -------- Chat Endpoint -------- #
@app.post("/chat/")
async def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        return {"response": "⚠️ Please enter a message."}

    try:
        if retriever_tools:
            # Take the first retriever
            source, data = list(retriever_tools.items())[0]
            retriever = data["retriever"]

            docs = retriever.get_relevant_documents(message)

            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                prompt = f"Context from {source}:\n{context}\n\nQuestion: {message}"
                response = llm.invoke(prompt).content
            else:
                response = "❌ Sorry, I couldn't find information about that in the uploaded documents."
        else:
            response = "⚠️ Please upload a document or provide a URL first."

        return {"response": response}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)




# -------- Debug Info Endpoint -------- #
@app.get("/vectorstore_info/")
async def vectorstore_info():
    info = {name: len(vs.index_to_docstore_id) for name, vs in vectorstores.items()}
    return {"sources": info}
