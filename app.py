"""
Rohit Virkud Second Brain - RAG-powered chat app.
FastAPI + fastembed + Anthropic Claude Haiku.
"""

import os
import glob
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
from anthropic import Anthropic

POSTS_DIR = os.environ.get("POSTS_DIR", "/home/letbu/RV_Second_Brain")

app = FastAPI(title="Rohit's Second Brain")

# --- Load posts ---
posts = []
post_files = sorted(glob.glob(os.path.join(POSTS_DIR, "[0-9]*.md")))
for f in post_files:
    with open(f) as fh:
        text = fh.read().strip()
    posts.append({"file": os.path.basename(f), "text": text})
print(f"Loaded {len(posts)} posts")

# --- Build embeddings ---
print("Loading embedding model...")
encoder = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = np.array(list(encoder.embed([p["text"] for p in posts])))
print(f"Embeddings shape: {embeddings.shape}")

# --- Anthropic client ---
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    env_path = os.path.join(POSTS_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                elif line.startswith("OPENAI_API_KEY="):
                    continue

llm = Anthropic(api_key=api_key) if api_key else None

PERSONA_PROMPT = """You are Rohit Virkud — a brand strategist who helps independent financial advisors build their practice on LinkedIn.

YOUR VOICE RULES:
- Speak directly. Use "you", "your". Be conversational, not corporate.
- Short paragraphs. Line breaks for punch. No long blocks.
- Be opinionated. You have strong takes on positioning, distribution over content, and flat fee vs AUM.
- Use real examples. Refer to clients, experiences, things that actually happened.
- Tactical before theory. Give actionable steps, not abstractions.
- Humble but confident. You've done the work. You know what works.
- If you don't know something from the context, say so directly.

YOUR CORE BELIEFS:
- Distribution > Content creation. The 15% people see isn't the whole game.
- Niche finds you through doing good work, not thinking about it.
- Flat fee pricing > AUM for financial advisors.
- Inbound works: connect with ICPs → post commercial content → DM engagers.
- Relationships and partnerships drive real business growth.

Answer the question using the LinkedIn posts provided as context. If the context doesn't cover the question, say "I haven't talked about that specifically, but here's what I'd say based on my experience..." and give your honest take.

CRITICAL FORMATTING RULE: Do NOT use asterisks (*) anywhere in your response. No bold, no italic, no bullet lists with asterisks. Use dashes (-) for lists and plain text for emphasis."""


class ChatRequest(BaseModel):
    message: str


def find_relevant_posts(query, top_k=5):
    query_vec = np.array(list(encoder.embed([query])))
    scores = np.dot(embeddings, query_vec.T).flatten()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    results = []
    for idx in top_idx:
        results.append({
            "score": float(scores[idx]),
            "file": posts[idx]["file"],
            "text": posts[idx]["text"],
        })
    return results


def build_prompt(query, context_posts):
    context = ""
    for i, p in enumerate(context_posts, 1):
        context += f"--- POST {i} ---\n{p['text']}\n\n"
    return f"""{PERSONA_PROMPT}

CONTEXT FROM ROHIT'S LINKEDIN POSTS:
{context}

QUESTION: {query}

ANSWER (in Rohit's voice, using the context above):"""


@app.get("/", response_class=HTMLResponse)
async def index():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rohit's Second Brain</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0a0a;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  .header {
    padding: 16px 24px;
    border-bottom: 1px solid #1a1a1a;
    background: #0d0d0d;
  }
  .header h1 { font-size: 18px; font-weight: 600; color: #fff; }
  .header p { font-size: 13px; color: #666; margin-top: 2px; }
  .chat {
    flex: 1; overflow-y: auto; padding: 24px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .msg {
    max-width: 680px; padding: 14px 18px; border-radius: 10px;
    line-height: 1.6; font-size: 14px;
  }
  .msg.user {
    background: #1a3a2a; color: #b8e6c8;
    align-self: flex-end; border-bottom-right-radius: 4px;
  }
  .msg.bot {
    background: #141414; color: #d0d0d0;
    align-self: flex-start; border-bottom-left-radius: 4px;
    border: 1px solid #1f1f1f;
  }
  .input-area {
    padding: 16px 24px; border-top: 1px solid #1a1a1a;
    background: #0d0d0d;
  }
  .input-row {
    display: flex; gap: 10px; max-width: 720px; margin: 0 auto;
  }
  .input-row input {
    flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid #222;
    background: #111; color: #e0e0e0; font-size: 14px; outline: none;
  }
  .input-row input:focus { border-color: #4a8; }
  .input-row input::placeholder { color: #444; }
  .input-row button {
    padding: 12px 24px; border-radius: 8px; border: none;
    background: #1a6a3a; color: #fff; font-size: 14px; font-weight: 500;
    cursor: pointer; transition: background 0.2s;
  }
  .input-row button:hover { background: #228a4a; }
  .input-row button:disabled { background: #333; cursor: not-allowed; }
  .typing { color: #555; font-size: 13px; padding: 8px 18px; }
  pre { white-space: pre-wrap; font-family: inherit; margin: 0; }
  @media (max-width: 600px) {
    .chat { padding: 16px; }
    .msg { max-width: 100%; font-size: 13px; }
    .input-area { padding: 12px 16px; }
  }
</style>
</head>
<body>
<div class="header">
  <h1>Rohit's Second Brain</h1>
  <p>Trained on 50 LinkedIn posts · Ask anything like you'd ask Rohit</p>
</div>
<div class="chat" id="chat">
  <div class="msg bot">
    Hey — ask me anything about LinkedIn, positioning, content, or building a practice. I'll answer like I would on a call.
  </div>
</div>
<div class="input-area">
  <div class="input-row">
    <input type="text" id="input" placeholder="Ask Rohit..." autofocus>
    <button id="sendBtn" onclick="send()">Send</button>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const btn = document.getElementById('sendBtn');

input.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') send();
});

function addMessage(text, role) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<pre>' + escapeHtml(text) + '</pre>';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'msg bot typing';
  div.id = 'typing';
  div.textContent = 'thinking...';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

async function send() {
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  btn.disabled = true;
  addMessage(msg, 'user');
  addTyping();
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg})
    });
    const data = await res.json();
    removeTyping();
    addMessage(data.answer, 'bot');
  } catch(e) {
    removeTyping();
    addMessage('Error: ' + e.message, 'bot');
  }
  btn.disabled = false;
  input.focus();
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/chat")
async def chat(req: ChatRequest):
    if not llm:
        return JSONResponse(
            status_code=500,
            content={"error": "ANTHROPIC_API_KEY not set. Create a .env file in RV_Second_Brain/ with ANTHROPIC_API_KEY=sk-ant-..."}
        )

    query = req.message.strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "Empty message"})

    relevant = find_relevant_posts(query, top_k=4)
    prompt = build_prompt(query, relevant)

    try:
        resp = llm.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = resp.content[0].text.strip()
        answer = answer.replace("*", "")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return JSONResponse({"answer": answer})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
