# app.py
"""
StudyMate — PDF-aware AI study assistant (Gradio).
Features:
 - Upload multiple PDFs -> extract text (PyMuPDF)
 - Build embeddings (sentence-transformers) + FAISS index (fast retrieval)
 - Use retrieved chunks as context for an LLM to answer user questions
 - Chat UI with animations, chat history, download conversation
 - Attempts IBM Granite if HF_TOKEN is provided, falls back to google/flan-t5-small
"""

import os
import math
import textwrap
import gradio as gr
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
import faiss
import fitz  # PyMuPDF
from html import escape

# ---------------------------
# CONFIG
# ---------------------------
# Optionally set HF_TOKEN in your environment if you have access to IBM Granite
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# IBM Granite model name (change if you have a different Granite repo)
GRANITE_MODEL = "ibm-granite/granite-3.3-2b-instruct"

# Fallback small, fast instruction model (works without gating)
FALLBACK_TEXT2TEXT = "google/flan-t5-small"  # text2text-generation

# Embedding model (small & fast)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval settings
CHUNK_SIZE = 800       # characters per chunk (~ short passages)
CHUNK_STRIDE = 200     # overlap between chunks
EMBED_BATCH = 32       # batch size for embedding
TOP_K = 5              # number of relevant chunks to retrieve per query
MAX_PROMPT_TOKENS = 1024  # guard for prompt length; don't exceed model limits

# ---------------------------
# UTILS: PDF extraction, chunking, embedding, FAISS index
# ---------------------------
def extract_text_from_pdf(path):
    """Extract text from a PDF file path using PyMuPDF."""
    try:
        doc = fitz.open(path)
        full_text = []
        for page in doc:
            txt = page.get_text("text")
            if txt:
                full_text.append(txt)
        doc.close()
        return "\n".join(full_text).strip()
    except Exception as e:
        return f"[Error reading {path}: {e}]"

def chunk_text(text, chunk_size=CHUNK_SIZE, stride=CHUNK_STRIDE):
    """Split text into overlapping chunks."""
    text = text.replace("\r", "")
    start = 0
    chunks = []
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start += chunk_size - stride
    return chunks

# Initialize embedding model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts):
    """Return embeddings (numpy array) for a list of texts."""
    # SentenceTransformer returns numpy arrays
    return embed_model.encode(texts, batch_size=EMBED_BATCH, show_progress_bar=False, convert_to_numpy=True)

# A small helper to build & manage FAISS index for multiple files
class Retriever:
    def __init__(self):
        self.index = None
        self.texts = []   # chunk texts in same order as index
        self.meta = []    # metadata (file names, chunk idx)
        self.dim = embed_model.get_sentence_embedding_dimension()

    def add_texts(self, docs_with_meta):
        """
        docs_with_meta: iterable of tuples (text, meta_dict)
        meta_dict can contain 'source' or page info
        """
        new_texts = []
        new_meta = []
        for txt, m in docs_with_meta:
            new_texts.append(txt)
            new_meta.append(m)
        if not new_texts:
            return
        embs = embed_texts(new_texts)
        if self.index is None:
            # create index
            self.index = faiss.IndexFlatIP(self.dim)  # cosine similarity via normalized vectors -> inner product
            # normalize vectors
            faiss.normalize_L2(embs)
            self.index.add(embs)
        else:
            faiss.normalize_L2(embs)
            self.index.add(embs)
        self.texts.extend(new_texts)
        self.meta.extend(new_meta)

    def search(self, query, top_k=TOP_K):
        """Return top_k (text, meta, score) for a query string."""
        if self.index is None or len(self.texts) == 0:
            return []
        q_emb = embed_texts([query])
        faiss.normalize_L2(q_emb)
        scores, ids = self.index.search(q_emb, top_k)
        results = []
        for sid, sc in zip(ids[0], scores[0]):
            if sid < 0:
                continue
            results.append((self.texts[sid], self.meta[sid], float(sc)))
        return results

# ---------------------------
# MODEL LOADING (Generation)
# ---------------------------
def load_generator():
    """
    Try to load Granite if HF_TOKEN available, else load a fallback text2text model.
    For Granite (causal), we use text-generation pipeline.
    For FLAN (text2text), we use text2text-generation pipeline.
    """
    # Prefer Granite if token present
    if HF_TOKEN:
        try:
            print(f"Attempting to load Granite model: {GRANITE_MODEL}")
            gen = pipeline(
                "text-generation",
                model=GRANITE_MODEL,
                use_auth_token=HF_TOKEN,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            print("Loaded Granite successfully.")
            return {"pipe": gen, "mode": "causal", "name": GRANITE_MODEL}
        except Exception as e:
            print("Granite load failed (will fallback):", e)

    # Fallback to text2text (FLAN T5 small)
    print(f"Loading fallback text2text model: {FALLBACK_TEXT2TEXT}")
    gen = pipeline("text2text-generation", model=FALLBACK_TEXT2TEXT, device_map="auto" if torch.cuda.is_available() else None)
    print("Loaded fallback model.")
    return {"pipe": gen, "mode": "text2text", "name": FALLBACK_TEXT2TEXT}

gen_info = load_generator()
gen_pipe = gen_info["pipe"]
GEN_MODE = gen_info["mode"]

def generate_answer_with_context(question, context_chunks, max_tokens=200, temperature=0.7, top_p=0.9):
    """
    Build prompt from context_chunks (list of texts) + question, then call the generator pipeline.
    Returns plain reply string.
    """
    # Compose helpful system/instruction + contexts
    instruction = (
        "You are StudyMate, an AI assistant. Use the provided context snippets from the uploaded PDFs to answer the user's question precisely and concisely. "
        "If the answer is not present in context, say you couldn't find it and provide general guidance if possible.\n\n"
    )

    # Join top context chunks with separators
    joined_ctx = "\n\n---\n\n".join(context_chunks).strip()

    # Build prompt depending on generator mode
    if GEN_MODE == "text2text":
        # For text2text models (FLAN), provide clear instruction + context + question
        prompt = f"{instruction}Context:\n{joined_ctx}\n\nQuestion:\n{question}\n\nAnswer:"
        out = gen_pipe(prompt, max_length=int(max_tokens), do_sample=True, temperature=float(temperature), top_p=float(top_p))
        text = out[0]["generated_text"]
        # strip any leading 'Answer:' etc.
        return text.strip()
    else:
        # causal model pipeline
        prompt = f"{instruction}Context:\n{joined_ctx}\n\nUser: {question}\nAssistant:"
        out = gen_pipe(prompt, max_new_tokens=int(max_tokens), do_sample=True, temperature=float(temperature), top_p=float(top_p))
        raw = out[0].get("generated_text", "")
        # attempt to trim prompt
        if raw.startswith(prompt):
            return raw[len(prompt):].strip()
        elif "Assistant:" in raw:
            return raw.split("Assistant:")[-1].strip()
        else:
            return raw.strip()

# ---------------------------
# GRADIO UI handlers
# ---------------------------
retriever = Retriever()  # global retriever for session (resets when server restarts)

def handle_upload(pdf_files):
    """
    Called when user uploads PDFs. We extract, chunk, embed, and add to FAISS index.
    Returns a status message for the UI.
    """
    if not pdf_files:
        return "No PDF uploaded."
    docs = []
    for f in pdf_files:
        # gradio File object has .name as path
        path = f.name if hasattr(f, "name") else f
        text = extract_text_from_pdf(path)
        if not text:
            continue
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            meta = {"source": os.path.basename(path), "chunk_id": i}
            docs.append((c, meta))
    if not docs:
        return "No text extracted from uploaded PDFs."
    retriever.add_texts(docs)
    return f"Indexed {len(docs)} chunks from {len(pdf_files)} file(s). You can now ask questions."

def handle_query(user_input, history, max_tokens, temperature, top_p):
    """
    Main chat handler: retrieve top chunks, call generator, update history and produce HTML.
    """
    if history is None:
        history = []
    # Retrieve relevant chunks
    # If no docs indexed, warn user
    if retriever.index is None or len(retriever.texts) == 0:
        reply = "No PDF uploaded yet — please upload PDF(s) first so I can answer from them."
        history.append((user_input, reply))
        return build_chat_html(history), history

    # retrieve top K relevant chunks
    hits = retriever.search(user_input, top_k=TOP_K)
    context_chunks = [h[0] for h in hits]
    # for debugging you can include metadata / scores

    # generate answer
    reply = generate_answer_with_context(user_input, context_chunks, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    history.append((user_input, reply))
    return build_chat_html(history), history

def build_chat_html(history):
    """Build chat HTML with animated bubbles from history list."""
    pieces = []
    for user, bot in history:
        u = escape(user).replace("\n", "<br>")
        b = escape(bot).replace("\n", "<br>")
        pieces.append(f"<div class='message user'><div class='meta'>You</div><div class='text'>{u}</div></div>")
        pieces.append(f"<div class='message bot'><div class='meta'>StudyMate</div><div class='text'>{b}</div></div>")
    html = "<div class='chat-box'>\n" + "\n".join(pieces) + "\n</div>"
    return html

# Download chat helper
def download_history(history):
    if not history:
        return None
    filename = "study_conversation.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for u, a in history:
            f.write("You: " + u + "\n\n")
            f.write("StudyMate: " + a + "\n\n")
    return filename

# ---------------------------
# GRADIO APP UI
# ---------------------------
css = """
.chat-box { background: #fff; padding: 18px; border-radius:14px; height:420px; overflow-y:auto; box-shadow:0 8px 30px rgba(7,14,31,0.08); }
.message { margin:10px 0; padding:10px 14px; display:block; max-width:82%; word-wrap:break-word; animation: fadeInMove 0.55s ease both; border-radius:12px; }
.message .meta { font-size:11px; opacity:0.75; margin-bottom:6px; font-weight:600; }
.message .text { white-space:pre-wrap; }
.message.user { background: linear-gradient(90deg,#0072ff,#00c6ff); color:white; margin-left:18%; float:right; border-radius:12px 12px 0 12px; }
.message.bot { background:#f4f7fb; color:#0f1724; margin-right:18%; float:left; border-radius:12px 12px 12px 0; }
@keyframes fadeInMove { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }
.controls { display:flex; gap:12px; align-items:center; }
.small { width:160px; }
.title { text-align:center; margin-bottom:12px; font-size:22px; font-weight:800; }
.header-sub { text-align:center; margin-bottom:18px; color: #374151; }
"""

with gr.Blocks(css=css, title="StudyMate AI") as demo:
    gr.HTML("<div class='title'>📚 StudyMate — AI study assistant</div>")
    gr.HTML("<div class='header-sub'>Upload PDFs, then ask questions — StudyMate will answer using your uploaded files.</div>")

    with gr.Row():
        with gr.Column(scale=3):
            uploaded = gr.File(label="Upload PDFs (multiple)", file_count="multiple", file_types=[".pdf"])
            upload_btn = gr.Button("⬆️ Index uploaded PDFs")
            upload_status = gr.Markdown("No PDFs indexed yet.")
            # Controls
            with gr.Row(elem_id="controls"):
                max_tokens = gr.Slider(50, 800, value=200, step=10, label="Max tokens", elem_classes="small")
                temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Temperature", elem_classes="small")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p", elem_classes="small")
            download_btn = gr.Button("🔽 Download Chat")
            clear_btn = gr.Button("🧹 Clear Chat")
            gr.Markdown("**Tips:** Use the sliders to control answer length & creativity. Upload PDFs first.")

        with gr.Column(scale=5):
            chat_html = gr.HTML(build_chat_html([]))
            msg = gr.Textbox(placeholder="Type your question here...", label="Ask a question")
            state = gr.State(value=[])

    # Wire up actions
    upload_btn.click(fn=lambda files: handle_upload(files) if files else "No files selected.", inputs=[uploaded], outputs=[upload_status])
    # main submit by pressing Enter
    msg.submit(fn=handle_query, inputs=[msg, state, max_tokens, temperature, top_p], outputs=[chat_html, state])
    # alternative button to send
    send_btn = gr.Button("Send")
    send_btn.click(fn=handle_query, inputs=[msg, state, max_tokens, temperature, top_p], outputs=[chat_html, state])

    clear_btn.click(fn=lambda: (build_chat_html([]), []), outputs=[chat_html, state])
    download_btn.click(fn=download_history, inputs=[state], outputs=[gr.File()])

    gr.HTML("<div style='text-align:center; font-size:12px; color:#6b7280; margin-top:12px;'>Model: " + gen_info["name"] + "</div>")

demo.launch()