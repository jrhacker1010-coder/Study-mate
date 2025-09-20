# =============================
# 📦 Install dependencies
# =============================
!pip install -q gradio transformers sentence-transformers faiss-cpu PyMuPDF

# =============================
# 🧠 Import libraries
# =============================
import os
import gradio as gr
import faiss
import fitz  # PyMuPDF
import torch
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# =============================
# ⚙️ Config
# =============================
HF_TOKEN = os.environ.get("HF_TOKEN", None)

GRANITE_MODEL = "ibm-granite/granite-3.3-2b-instruct"
FALLBACK_MODEL = "google/flan-t5-small"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_STRIDE = 200
TOP_K = 5

# =============================
# 📄 PDF text extractor
# =============================
def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        text = []
        for page in doc:
            txt = page.get_text("text")
            if txt:
                text.append(txt)
        doc.close()
        return "\n".join(text)
    except Exception as e:
        return f"[Error extracting text: {e}]"

def chunk_text(text, size=CHUNK_SIZE, stride=CHUNK_STRIDE):
    chunks, start, length = [], 0, len(text)
    while start < length:
        end = min(start + size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start += size - stride
    return chunks

# =============================
# 🔎 Embedding + FAISS Retriever
# =============================
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

class Retriever:
    def __init__(self):
        self.index = None
        self.texts = []
        self.meta = []
        self.dim = embed_model.get_sentence_embedding_dimension()

    def add_texts(self, docs):
        texts, meta = zip(*docs)
        embs = embed_model.encode(list(texts), convert_to_numpy=True)
        faiss.normalize_L2(embs)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)
        self.texts.extend(texts)
        self.meta.extend(meta)

    def search(self, query, k=TOP_K):
        if self.index is None:
            return []
        q_emb = embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, ids = self.index.search(q_emb, k)
        results = []
        for i, s in zip(ids[0], scores[0]):
            if i >= 0:
                results.append((self.texts[i], self.meta[i], float(s)))
        return results

retriever = Retriever()

# =============================
# 🤖 Load generator (LLM)
# =============================
def load_generator():
    if HF_TOKEN:
        try:
            gen = pipeline("text-generation", model=GRANITE_MODEL, use_auth_token=HF_TOKEN)
            return gen, "causal", GRANITE_MODEL
        except:
            pass
    gen = pipeline("text2text-generation", model=FALLBACK_MODEL)
    return gen, "text2text", FALLBACK_MODEL

gen_pipe, GEN_MODE, MODEL_NAME = load_generator()

# =============================
# 💬 Answer generator
# =============================
def generate_answer(question, context_chunks):
    instruction = "You are StudyMate, an AI study assistant. Use the context below to answer clearly.\n\n"
    joined_ctx = "\n\n---\n\n".join(context_chunks)

    if GEN_MODE == "text2text":
        prompt = f"{instruction}Context:\n{joined_ctx}\n\nQuestion: {question}\nAnswer:"
        out = gen_pipe(prompt, max_length=200)
        return out[0]["generated_text"].strip()
    else:
        prompt = f"{instruction}Context:\n{joined_ctx}\n\nUser: {question}\nAssistant:"
        out = gen_pipe(prompt, max_new_tokens=200)
        text = out[0]["generated_text"]
        return text.split("Assistant:")[-1].strip()

# =============================
# 🖥️ Gradio Handlers
# =============================
def handle_upload(files):
    docs = []
    if not files:
        return "⚠️ No files uploaded."

    for f in files:
        path = f.name if hasattr(f, "name") else f  # works for both file objects & paths
        try:
            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)
            for i, c in enumerate(chunks):
                docs.append((c, {"source": os.path.basename(path), "chunk": i}))
        except Exception as e:
            return f"❌ Error reading {path}: {e}"

    if not docs:
        return "⚠️ No text extracted."

    retriever.add_texts(docs)
    return f"✅ Indexed {len(docs)} chunks from {len(files)} file(s)."

def handle_query(q, history):
    if not history:
        history = []
    hits = retriever.search(q)
    if not hits:
        ans = "⚠️ No PDFs uploaded yet. Please upload first."
    else:
        context = [h[0] for h in hits]
        ans = generate_answer(q, context)
    history.append((q, ans))
    return history, history

# =============================
# 🎨 Gradio UI
# =============================
with gr.Blocks() as demo:
    gr.Markdown("## 📚 StudyMate — AI Study Assistant")
    with gr.Row():
        with gr.Column():
            pdfs = gr.File(label="Upload PDFs", file_types=[".pdf"], file_count="multiple")
            upload_btn = gr.Button("Index PDFs")
            status = gr.Textbox(label="Status")
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Ask a Question")
            state = gr.State([])

    upload_btn.click(handle_upload, inputs=[pdfs], outputs=[status])
    msg.submit(handle_query, inputs=[msg, state], outputs=[chatbot, state])

demo.launch(share=True)
