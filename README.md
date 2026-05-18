# 🎥 Vision-Talk

> **Intelligent Video Analysis Platform** — Describe, Search, and Query your video content using multimodal AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask) ![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎬 **Video Analysis** | Upload any video and extract AI-generated captions at configurable frame intervals |
| 📷 **Live Webcam** | Real-time frame analysis with on-screen caption overlay and session recording |
| 🔍 **Semantic Search** | Query your video library using natural language — powered by sentence embeddings + cross-encoder reranking |
| 💬 **Visual Q&A** | Ask questions about your videos and get context-aware answers from multimodal LLMs |
| 📚 **Video Library** | Browse, filter, and preview all processed videos with thumbnail extraction |
| 🔐 **Auth System** | Single-user mode with password/recovery-code login, email notifications, and profile management |

---

## 🤖 Supported AI Models

| Model | Provider | Caption | Q&A |
|---|---|:---:|:---:|
| **Gemini Flash 3** | Google | ✅ | ✅ |
| **NeMoVision** (12B) | NVIDIA | ✅ | ✅ |
| **Llama 3.2 90B Vision** | NVIDIA / Meta | ✅ | ✅ |
| **Llama 3.2 11B Vision** | NVIDIA / Meta | ✅ | ✅ |
| **Moondream** | Moondream | ✅ | ❌ |

---

## 🏗️ Architecture

```
vision-talk/
├── app.py                  # Flask + SocketIO entry point
├── config.py               # Paths, model configs, parameters
├── modules/
│   ├── base_analyzer.py    # Shared model loading & caption/QA logic
│   ├── watcher.py          # Uploaded video processing
│   ├── realtime.py         # Live webcam analysis + ffmpeg recording
│   ├── memory.py           # ChromaDB vector store (embeddings + reranking)
│   └── database.py         # SQLite auth (users, sessions, reset codes)
├── templates/
│   ├── index.html          # Main SPA (tabs: video / webcam / library / search / Q&A)
│   ├── login.html
│   ├── register.html
│   ├── profile.html
│   ├── forgot_password.html
│   ├── reset_with_code.html
│   └── email_templates.html
├── static/                 # CSS, JS, background video
├── uploads/                # Uploaded & recorded videos
├── downloads/              # Temp exports (cleared on startup)
├── chroma_db/              # Persistent vector database
└── vision_talk.db          # SQLite user database
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) in your system PATH (required for webcam recording)
- API key(s) for at least one supported model

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/vision-talk.git
cd vision-talk

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys (see below)

# 5. Run the application
python app.py
```

Open your browser at **http://localhost:5000**

On first launch, the registration page is accessible to create your account.

---

## ⚙️ Environment Variables

Create a `.env` file at the project root:

```env
# Google Gemini
GOOGLE_API_KEY=your_gemini_api_key

# Moondream
MOONDREAM_API_KEY=your_moondream_api_key

# NVIDIA (for NeMoVision, Llama 90B/11B Vision)
NVIDIA_API_KEY=nvapi-your_nvidia_api_key
```

> API keys can also be added per-model directly from your **Profile** page inside the app.

---

## 📋 Requirements

```
flask
flask-socketio
werkzeug
python-dotenv
opencv-python
Pillow
chromadb
sentence-transformers
google-genai
moondream
requests
numpy
```

> Generate a full `requirements.txt` with: `pip freeze > requirements.txt`

---

## 🔧 Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `FRAME_INTERVAL` | `5` | Seconds between analyzed frames |
| `QA_MAX_SEGMENTS` | `15` | Max segments retrieved for Q&A context |
| `RERANKING_MULTIPLIER` | `3` | Initial over-fetch factor for cross-encoder reranking |
| `REALTIME_MAX_QUEUE_SIZE` | `5` | Frame queue depth for webcam analyzer |
| `GEMINI_MODEL` | `models/gemini-3-flash-preview` | Gemini model identifier |

> The embedding and reranking models are loaded from local HuggingFace cache paths — update these in `config.py` to match your environment or switch to model name strings for automatic download.

---

## 🖥️ Usage

### 1. Upload & Analyze a Video
1. Upload a video file via the sidebar (drag & drop or click)
2. Select an AI model
3. Adjust the frame interval slider
4. Click **Process Video** — captions are stored in the vector DB in real time

### 2. Live Webcam Analysis
1. Switch to the **Live Webcam** tab
2. Select model and camera
3. Click **Start Recording** — captions appear as an overlay
4. Click **Stop** to save the session as an MP4

### 3. Semantic Search
- Type a natural language query in the **Semantic Search** tab
- Results are ranked by cosine similarity + cross-encoder reranking
- Click any result to jump to that moment in the video

### 4. Visual Q&A
- Switch to the **Q&A** tab
- Type your question (requires a model that supports Q&A)
- The system retrieves the most relevant video segments and generates a grounded answer

---

## 🔐 Authentication

Vision-Talk runs in **single-user mode** — only one account can be registered per instance.

- Login with **password** or your **6-digit permanent recovery code**
- Forgot password? Use the email-based reset flow (temporary 6-digit code, expires in 10 min)
- A new permanent recovery code is issued every time your password changes

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [sentence-transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` for embeddings, `ms-marco-MiniLM-L-6-v2` for reranking
- [ChromaDB](https://www.trychroma.com/) — vector store
- [Google Gemini](https://ai.google.dev/) — multimodal LLM
- [NVIDIA NIM](https://build.nvidia.com/) — NeMoVision & Llama Vision inference
- [Moondream](https://moondream.ai/) — lightweight VLM
