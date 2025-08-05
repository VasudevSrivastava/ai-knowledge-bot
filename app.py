import tempfile
from google import genai
import fitz  # PyMuPDF
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os

if "PINECONE_API_KEY" in st.secrets:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


index_name = "knowledge-bot"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if not pc.has_index(index_name):

    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development"
        }
    )

index = pc.Index(index_name)
index.delete(delete_all=True)

st.set_page_config(page_title="AI Knowledge Bot", page_icon="🤖")

st.title("📄 AI Knowledge Bot")
st.write("Upload a PDF and ask questions about its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", size_limit=10 * 1024 * 1024)  # 10 MB limit

raw_text = ""

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    def load_pdf(file_path):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    raw_text = load_pdf(file_path)

  

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)




if st.button("Process Document"):
    chunks = split_text(raw_text)

    embeddings = embed_model.encode(chunks, show_progress_bar=True)

    vectors = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        vector_id = f"chunk-{i}"
        metadata = {"text": chunk}
        vectors.append((vector_id, vector.tolist(), metadata))

    index.upsert(vectors=vectors)
    st.success("Data stored in Pinecone!")


@st.cache_resource
def get_genai_client():
    return genai.Client()

client = get_genai_client()


st.markdown("---")
st.subheader("Ask a Question")

query = st.text_input("Enter your question:")
if st.button("Get Answer") and query:
    if not uploaded_file:
        st.error("Please upload a PDF file first.")
        st.stop()
    else:
        st.write("Processing your question...")
        query_vector = embed_model.encode([query])[0]

        result = index.query(vector=query_vector.tolist(), top_k=5, include_metadata=True)
        retrieved_texts = [match['metadata']['text'] for match in result['matches']]

        context = "\n\n".join(retrieved_texts)

        prompt = f"""Answer the following question using only the context below.

Context:
{context}

Question: {query}

Answer:"""

    client = get_genai_client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.markdown("### 🤖 Answer:")
    st.write(response.text)


