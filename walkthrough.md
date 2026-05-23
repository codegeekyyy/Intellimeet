# Backend Build Walkthrough
## Meeting Summarizer & Interview Prep Platform

> **Stack:** Python · FastAPI · PostgreSQL · Redis · BullMQ · Groq API · pyannote · faster-whisper
> **Phase:** 1 — Foundation build (free-tier hosted APIs throughout)
> **Audience:** Backend engineers, DevOps, technical leads

---

## Table of Contents

1. [Project Structure & Environment Setup](#1-project-structure--environment-setup)
2. [Database Layer — PostgreSQL + Redis](#2-database-layer)
3. [Audio Ingestion Service — File Upload](#3-audio-ingestion-service)
4. [Audio Pre-processing — FFmpeg + pydub](#4-audio-pre-processing)
5. [Source Separation — SpeechBrain SepFormer](#5-source-separation)
6. [Speech-to-Text — faster-whisper (Free, Local)](#6-speech-to-text)
7. [Speaker Diarization — pyannote.audio](#7-speaker-diarization)
8. [Job Queue — BullMQ + Redis Workers](#8-job-queue)
9. [LLM Reasoning Layer — Groq API (Free)](#9-llm-reasoning-layer)
10. [Meeting Summary Output & Delivery](#10-meeting-summary-output--delivery)
11. [Interview Prep Pipeline](#11-interview-prep-pipeline)
12. [FastAPI Backend — Full Route Map](#12-fastapi-route-map)
13. [Authentication & Security](#13-authentication--security)
14. [Infrastructure & Deployment](#14-infrastructure--deployment)
15. [Environment Variables & Secrets](#15-environment-variables--secrets)
16. [Testing Strategy](#16-testing-strategy)

---

## Architecture Overview

```mermaid
graph TB
    subgraph Client
        A["Browser / Mobile"]
    end

    subgraph "FastAPI Server"
        B["HTTP API<br/>(upload, CRUD, auth)"]
    end

    subgraph Storage
        D["PostgreSQL<br/>(persistent data)"]
        E["Redis<br/>(queues, cache, pubsub)"]
        F["Local Filesystem<br/>(audio files)"]
    end

    subgraph "Audio Worker (GPU)"
        G["FFmpeg + pydub<br/>(preprocess)"]
        H["SpeechBrain SepFormer<br/>(source separation)"]
        I["faster-whisper<br/>(STT)"]
        J["pyannote.audio<br/>(diarization)"]
    end

    subgraph "LLM Worker"
        K["Groq API<br/>(Llama 3.1 70B — FREE)"]
    end

    A -->|upload| B
    B --> F
    B --> E
    E -->|BullMQ job| G
    G --> H --> I --> J
    J -->|transcript| E
    E -->|BullMQ job| K
    K -->|summary JSON| D
    B --> D
```

> [!IMPORTANT]
> **All external APIs used are free tier:** Groq (LLM), HuggingFace (pyannote model), faster-whisper (local, MIT). No credit card required for development. Audio files are stored on the **local filesystem** — no cloud storage dependency.

---

## 1. Project Structure & Environment Setup

### Monorepo Layout

```
platform/
  backend/
    app/
      main.py              # FastAPI entry point
      config.py            # Settings (pydantic-settings)
      database.py          # SQLAlchemy async engine
      models/
        user.py            # SQLAlchemy ORM models
        session.py
        transcript.py
        summary.py
      routers/
        audio.py           # Upload endpoint
        meetings.py        # Summary CRUD
        interview.py       # Interview mode endpoints
        auth.py            # JWT auth
      schemas/
        audio.py           # Pydantic I/O schemas
        summary.py
      services/
        storage.py         # Local filesystem file ops
        queue.py           # BullMQ job dispatch
  workers/
    audio_processor/
      worker.py            # FFmpeg + chunking
      separator.py         # SpeechBrain SepFormer
      diarizer.py          # pyannote.audio
      transcriber.py       # faster-whisper
      pipeline.py          # Orchestrates all steps
    llm_processor/
      worker.py            # Groq API calls
      prompts.py           # Prompt templates
      parser.py            # Output JSON parsing
  infra/
    docker-compose.yml
    postgres/init.sql
    redis/redis.conf
  tests/
    test_pipeline.py
    test_api.py
  .env.example
  requirements.txt
  Dockerfile.api
  Dockerfile.worker
```

> [!NOTE]
> The backend is split into **three distinct services** — the FastAPI HTTP server, the audio processing worker, and the LLM reasoning worker. Each runs as its own Docker container. They communicate via Redis queues (BullMQ jobs) and share a single PostgreSQL database. Audio files live on a **shared local volume** mounted into every container that needs access.

### Environment Setup

```bash
# 1. Create virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install all dependencies
pip install fastapi uvicorn[standard] sqlalchemy[asyncio] asyncpg alembic \
  pydantic-settings python-jose[cryptography] passlib[bcrypt] \
  redis bullmq python-multipart aiofiles \
  pydub ffmpeg-python speechbrain pyannote.audio \
  faster-whisper openai httpx pytest pytest-asyncio

# 3. Copy and fill environment variables
cp .env.example .env

# 4. Start infrastructure (Postgres + Redis)
docker compose up -d postgres redis

# 5. Run migrations
alembic upgrade head

# 6. Start API server
uvicorn app.main:app --reload --port 8000

# 7. Start workers (separate terminals)
python workers/audio_processor/worker.py
python workers/llm_processor/worker.py
```

---

## 2. Database Layer

### PostgreSQL — Persistent Data

SQLAlchemy with asyncpg provides async database access.

```python
# app/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/platform"

engine = create_async_engine(DATABASE_URL, pool_size=20, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession,
                                 expire_on_commit=False)
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

### Core ORM Models

```python
# app/models/session.py
class MeetingSession(Base):
    __tablename__ = "meeting_sessions"
    id          = Column(UUID, primary_key=True, default=uuid.uuid4)
    user_id     = Column(UUID, ForeignKey("users.id"), nullable=False)
    title       = Column(String(255))
    status      = Column(Enum("queued","processing","complete","failed"))
    audio_path  = Column(String(512))        # local filesystem path
    duration_s  = Column(Float)
    created_at  = Column(DateTime, default=datetime.utcnow)

class Transcript(Base):
    __tablename__ = "transcripts"
    id          = Column(UUID, primary_key=True, default=uuid.uuid4)
    session_id  = Column(UUID, ForeignKey("meeting_sessions.id"))
    segments    = Column(JSONB)   # [{speaker, start, end, text, confidence}]
    raw_wer     = Column(Float)

class Summary(Base):
    __tablename__ = "summaries"
    id            = Column(UUID, primary_key=True, default=uuid.uuid4)
    session_id    = Column(UUID, ForeignKey("meeting_sessions.id"))
    overview      = Column(Text)
    decisions     = Column(JSONB)   # [{text, speaker, timestamp}]
    action_items  = Column(JSONB)   # [{task, owner, deadline, priority}]
    open_questions= Column(JSONB)
    sentiment     = Column(JSONB)   # {speaker_id: score}
    model_used    = Column(String(100))
    tokens_used   = Column(String(50))
```

### Redis Key Schema

| Key Pattern | Purpose |
|---|---|
| `session:{id}:status` | Current processing status — queued / transcribing / summarising / done |
| `job:{id}:progress` | Integer 0–100 — pipeline progress for progress bar UI |
| `user:{id}:ratelimit` | Rolling window counter — API rate limiting per user |

### Alembic Migrations

```bash
alembic revision --autogenerate -m "add_interview_sessions_table"
alembic upgrade head      # Apply all pending
alembic downgrade -1      # Rollback one
alembic current           # Check current revision
```

---

## 3. Audio Ingestion Service

Audio files are uploaded via a standard multipart HTTP POST and saved directly to the **local filesystem**.

```mermaid
graph LR
    A["File Upload<br/>(multipart POST)"] --> B["Local Filesystem<br/>(UPLOAD_DIR)"]
    B --> C["BullMQ Job<br/>(audio queue)"]
```

### Local File Storage Service

```python
# app/services/storage.py
import os
import uuid
import aiofiles
from app.config import settings

async def save_audio_locally(tmp_path: str, user_id: str) -> str:
    """Move uploaded audio to the persistent upload directory.
    Returns the relative path within UPLOAD_DIR."""
    ext = os.path.splitext(tmp_path)[1]
    filename = f"{user_id}/{uuid.uuid4()}{ext}"
    dest = os.path.join(settings.upload_dir, filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # Move temp file into permanent location
    os.replace(tmp_path, dest)
    return filename

def get_audio_full_path(relative_path: str) -> str:
    """Resolve a stored relative path to an absolute filesystem path."""
    return os.path.join(settings.upload_dir, relative_path)

def delete_audio_file(relative_path: str) -> None:
    """Remove an audio file from disk."""
    full = get_audio_full_path(relative_path)
    if os.path.exists(full):
        os.remove(full)
```

### File Upload Endpoint

```python
# app/routers/audio.py
ALLOWED_MIME = {"audio/mpeg","audio/wav","audio/mp4","audio/x-m4a",
                "audio/ogg","video/mp4","video/webm"}
MAX_SIZE_MB = 500

@router.post("/upload")
async def upload_meeting(file: UploadFile, title: str = "Untitled Meeting",
                         db=Depends(get_db), user=Depends(get_current_user)):
    # Validate mime → Stream to temp file → Save locally → Create DB record → Dispatch job
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(400, f"Unsupported: {file.content_type}")

    # Stream to temp to avoid memory blow-up
    tmp_path = os.path.join(settings.upload_dir, "tmp", f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    async with aiofiles.open(tmp_path, "wb") as f:
        total = 0
        async for chunk in file:
            total += len(chunk)
            if total > MAX_SIZE_MB * 1024 * 1024:
                os.remove(tmp_path)
                raise HTTPException(413, "File too large")
            await f.write(chunk)

    audio_path = await save_audio_locally(tmp_path, str(user.id))
    session = MeetingSession(user_id=user.id, title=title,
                              audio_path=audio_path, status="queued")
    db.add(session); await db.commit()
    job_id = await dispatch_audio_job(str(session.id), audio_path)
    return {"session_id": str(session.id), "job_id": job_id, "status": "queued"}
```

---

## 4. Audio Pre-processing

Every audio file goes through: **normalization → silence trimming → format conversion (16kHz mono WAV) → chunking**.

```mermaid
graph LR
    A["Raw Audio<br/>(any format)"] --> B["FFmpeg<br/>16kHz mono WAV<br/>loudness norm"]
    B --> C["pydub<br/>silence trim"]
    C --> D["Chunker<br/>30s windows<br/>4s overlap"]
    D --> E["Chunks[]"]
```

| Parameter | Value | Reason |
|---|---|---|
| Sample rate | 16,000 Hz | Whisper requirement |
| Channels | 1 (mono) | Reduces processing time |
| Chunk size | 30 seconds | Optimal for Whisper context window |
| Overlap | 4 seconds | Prevents mid-word cuts at boundaries |
| Silence threshold | -40 dBFS | Removes dead air |

```python
def preprocess_audio(input_path: str) -> str:
    """Normalize, convert to 16kHz mono WAV via FFmpeg."""
    ffmpeg.input(input_path).output(out_path,
        ar=16000, ac=1, af="loudnorm", acodec="pcm_s16le"
    ).overwrite_output().run(quiet=True)

def chunk_audio(wav_path: str) -> list[dict]:
    """Slice into 30s windows with 4s overlap. Returns [{path, start_ms, end_ms}]."""
    # Step through audio at (30-4)=26s intervals
```

---

## 5. Source Separation — SpeechBrain SepFormer

When diarization detects **>15% overlapping speech**, SepFormer unmixes audio into separate speaker streams. Each stream is then transcribed independently.

```python
# Load model once at worker startup (~400MB download, free)
separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="/models/sepformer"
)

def separate_speakers(wav_path: str, num_speakers: int = 2) -> list[str]:
    """Separate mixed audio into individual speaker streams."""
    est_sources = separator.separate_file(path=wav_path)
    # Output at 8kHz → upsample back to 16kHz for Whisper
    # Graceful fallback: returns original file if separation fails

def should_separate(overlap_ratio: float) -> bool:
    return overlap_ratio > 0.15  # Only run if >15% overlap
```

> [!TIP]
> SepFormer is computationally expensive. The `should_separate()` gate ensures it only runs when overlapping speech is significant enough to warrant the cost.

---

## 6. Speech-to-Text — faster-whisper

| Property | Value |
|---|---|
| **Model** | large-v3 (3GB VRAM) / base for low-resource |
| **Speed on GPU** | ~4x faster than vanilla Whisper — 60-min meeting in ~4 min on A10G |
| **Speed on CPU** | 10–15 min for 60-min meeting |
| **Languages** | 99 languages — auto-detected or specified |
| **Licence** | MIT — free for commercial use |

```python
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def transcribe_chunk(wav_path: str, language=None) -> list[dict]:
    segments, info = model.transcribe(wav_path,
        beam_size=5, word_timestamps=True, vad_filter=True,
        condition_on_previous_text=True)
    # Returns [{word, start, end, probability}]

def deduplicate_words(words):
    """Remove duplicates in overlap zones using 50ms buckets."""
```

> [!NOTE]
> **No API key needed.** faster-whisper runs 100% locally. The deduplication step handles words that appear in the overlap zones between adjacent chunks.

---

## 7. Speaker Diarization — pyannote.audio

pyannote.audio 3.1 assigns each word a speaker label via voice embedding clustering.

```python
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                     use_auth_token=HF_TOKEN)  # free HuggingFace account

def diarize(wav_path, num_speakers=None) -> dict:
    """Returns {turns: [{speaker, start, end, overlap}], num_speakers: int}"""

def align_words_to_speakers(words, turns) -> list[dict]:
    """Assign each transcribed word to its speaker based on timestamp midpoint."""

def group_into_segments(words) -> list[dict]:
    """Group consecutive same-speaker words into utterance segments.
       Gap < 1.5s = same utterance."""
```

### Pipeline Flow: STT + Diarization Merge

```mermaid
graph LR
    A["Word-level<br/>timestamps"] --> C["Align"]
    B["Speaker<br/>turns"] --> C
    C --> D["Speaker-attributed<br/>segments"]
    D --> E["[{speaker, start, end, text, confidence, overlap}]"]
```

---

## 8. Job Queue — BullMQ + Redis

```mermaid
graph LR
    A["API Server"] -->|dispatch| B["Redis Queue<br/>'bull:audio:waiting'"]
    B -->|brpop| C["Audio Worker"]
    C -->|dispatch| D["Redis Queue<br/>'bull:llm:waiting'"]
    D -->|brpop| E["LLM Worker"]
    E -->|save| F["PostgreSQL"]
```

```python
# Dispatch jobs
async def dispatch_audio_job(session_id, audio_path) -> str:
    job_id = str(uuid.uuid4())
    await redis.lpush("bull:audio:waiting", json.dumps({
        "id": job_id,
        "session_id": session_id,
        "audio_path": audio_path,   # local filesystem path
    }))
    await redis.set(f"job:{job_id}:progress", 0, ex=86400)
    return job_id

# Worker loop (blocking pop)
async def main():
    while True:
        _, raw = await redis.brpop("bull:audio:waiting", timeout=0)
        job = json.loads(raw)
        await run_audio_pipeline(job)
```

### Full Pipeline Orchestration (8 steps)

| Step | Progress | Status | Action |
|---|---|---|---|
| 1 | 5% | downloading | Read audio from local disk |
| 2 | 15% | preprocessing | FFmpeg normalize + trim + chunk |
| 3 | 30% | diarizing | pyannote speaker turns |
| 4 | 40% | separating | SepFormer (if >15% overlap) |
| 5 | 55% | transcribing | faster-whisper all chunks |
| 6 | 75% | aligning | Map words → speakers |
| 7 | 85% | saving | Persist transcript to DB |
| 8 | 90% | queued_for_llm | Dispatch to LLM queue |

---

## 9. LLM Reasoning Layer — Groq API (Free)

| Property | Value |
|---|---|
| **API** | Groq — `api.groq.com/openai/v1` (OpenAI-compatible) |
| **Model** | llama-3.1-70b-versatile — free tier, 128K context |
| **Free limits** | 14,400 req/day · 30 RPM · 6,000 TPM output |
| **Cost** | $0 — fully free for dev and moderate production |
| **Key** | Free at console.groq.com — no credit card |

### Prompt Engineering

The system uses structured JSON prompts with explicit schema:

```python
SYSTEM_PROMPT = """You are an expert meeting analyst. You receive a speaker-attributed
meeting transcript and return a structured JSON summary. Be precise and factual.
Only extract what was explicitly said — do not infer or invent."""

# Output schema enforced:
{
  "overview": "2-3 sentence executive summary",
  "decisions": [{"text": "...", "speaker": "...", "timestamp": 12.4}],
  "action_items": [{"task": "...", "owner": "...", "deadline": "...", "priority": "high|medium|low"}],
  "open_questions": ["..."],
  "sentiment": {"overall": "positive|neutral|negative", "notes": "..."}
}
```

### Output Parser — Graceful Fallback

```python
def parse_summary_response(raw: str) -> dict:
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())  # Strip fences
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)  # Extract JSON substring
        data = json.loads(match.group()) if match else raise
    # Validate required keys, fill missing with defaults
```

---

## 10. Meeting Summary Output & Delivery

Two delivery channels:

| Channel | Method | Details |
|---|---|---|
| **API** | `GET /meetings/{id}/summary` | JSON response with full summary |
| **Email** | SendGrid free tier | 100 emails/day (optional) |

```python
@router.get("/{session_id}/summary")
async def get_summary(session_id: str):
    # Returns complete summary or current processing status

@router.get("/{session_id}/progress")
async def get_progress(session_id: str):
    # Returns {progress: 0-100, status: "transcribing"}
```

---

## 11. Interview Prep Pipeline

Reuses the same STT → LLM chain with **different prompts** + adds question generation and answer evaluation.

```mermaid
graph TB
    A["JD + Resume"] -->|generate| B["Tailored Questions<br/>(behavioral + technical)"]
    B --> C["User Records<br/>Spoken Answer"]
    C --> D["faster-whisper<br/>(transcribe, no diarization)"]
    D --> E["Groq LLM<br/>(evaluate answer)"]
    E --> F["Coaching Feedback"]
```

### Evaluation Output Schema

```json
{
  "score": 8,
  "star_method": {"situation": true, "task": true, "action": true, "result": false},
  "strengths": ["Clear problem definition", "Good technical depth"],
  "improvements": ["Add quantifiable results", "Mention team impact"],
  "model_answer_hint": "brief guide to a stronger answer",
  "clarity_score": 7,
  "relevance_score": 9
}
```

---

## 12. FastAPI Route Map

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Register user — returns JWT |
| `POST` | `/auth/login` | Login — returns access + refresh tokens |
| `POST` | `/auth/refresh` | Refresh access token |
| `POST` | `/audio/upload` | Upload meeting audio — dispatches job |
| `GET` | `/meetings/{id}/progress` | Poll job progress 0–100 |
| `GET` | `/meetings/{id}/summary` | Fetch completed summary JSON |
| `GET` | `/meetings/` | List all sessions for current user |
| `DELETE` | `/meetings/{id}` | Delete session + audio + summary |
| `POST` | `/interview/session/start` | Generate questions from JD + resume |
| `POST` | `/interview/session/{id}/answer` | Submit spoken answer — get feedback |
| `GET` | `/interview/session/{id}/history` | Get all attempts for a session |
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Prometheus metrics |

---

## 13. Authentication & Security

- **JWT-based** auth with access (30 min) + refresh (7 day) tokens
- **bcrypt** password hashing via passlib
- **OAuth2PasswordBearer** flow
- **HS256** signing algorithm

```python
def create_token(sub: str, ttl_minutes: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ttl_minutes)
    return jwt.encode({"sub": sub, "exp": expire}, SECRET_KEY, "HS256")

async def get_current_user(token = Depends(oauth2), db = Depends(get_db)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    user = await db.get(User, payload.get("sub"))
    if not user: raise HTTPException(401, "User not found")
    return user
```

---

## 14. Infrastructure & Deployment

### Docker Compose — Full Local Stack

| Service | Image | Notes |
|---|---|---|
| **postgres** | `pgvector/pgvector:pg16` | Persistent volume |
| **redis** | `redis:7-alpine` | 512MB max, LRU eviction |
| **api** | Custom `Dockerfile.api` | Port 8000, mounts `UPLOAD_DIR` volume |
| **audio-worker** | Custom `Dockerfile.worker` | **2 replicas**, GPU reserved, mounts `UPLOAD_DIR` volume |
| **llm-worker** | Custom `Dockerfile.worker` | **3 replicas** |

### Shared Volume for Audio Files

All containers that need access to uploaded audio mount the same Docker volume:

```yaml
# docker-compose.yml (excerpt)
volumes:
  audio-data:
    driver: local

services:
  api:
    volumes:
      - audio-data:/app/uploads
    environment:
      - UPLOAD_DIR=/app/uploads

  audio-worker:
    volumes:
      - audio-data:/app/uploads
    environment:
      - UPLOAD_DIR=/app/uploads
```

### Worker Dockerfile

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY workers/ ./workers/
COPY app/ ./app/
ENV WORKER_TYPE=audio
CMD ["sh", "-c", "python workers/${WORKER_TYPE}_processor/worker.py"]
```

---

## 15. Environment Variables & Secrets

| Variable | Source | Cost |
|---|---|---|
| `DATABASE_URL` | Local PostgreSQL | Free |
| `REDIS_URL` | Local Redis | Free |
| `JWT_SECRET_KEY` | Self-generated (64-char random) | Free |
| `UPLOAD_DIR` | Local filesystem path (e.g. `./uploads`) | Free |
| `GROQ_API_KEY` | console.groq.com | **Free** |
| `WHISPER_MODEL` | Local (large-v3 / base) | **Free (MIT)** |
| `HUGGINGFACE_TOKEN` | huggingface.co | **Free** |
| `SENDGRID_API_KEY` | Optional — 100 emails/day | Free tier |

```python
# app/config.py — typed settings with pydantic-settings
class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://localhost:6379"
    jwt_secret_key: str
    groq_api_key: str
    whisper_model: str = "large-v3"
    use_gpu: bool = True
    huggingface_token: str
    upload_dir: str = "./uploads"        # local audio storage root
    max_upload_size_mb: int = 500
    # ... all other vars
    class Config:
        env_file = ".env"
```

---

## 16. Testing Strategy

Three layers of tests:

| Layer | Target | GPU Needed |
|---|---|---|
| **Unit** | Individual pipeline steps (preprocess, chunk, parse) | No |
| **Integration** | Full audio → summary flow with 30s fixture | Yes |
| **API** | Every HTTP endpoint via `httpx.AsyncClient` | No |

```bash
pytest tests/ -v                                    # Run all
pytest tests/test_pipeline.py -v                    # Unit only (fast)
pytest tests/ --cov=app --cov=workers --cov-report=html  # With coverage
pytest tests/test_api.py -v                         # API only
```

### Key Test Cases

- `test_preprocess_produces_wav` — FFmpeg output is valid WAV
- `test_chunk_produces_overlapping_segments` — Overlap zones exist
- `test_transcribe_returns_words` — Word-level timestamps present
- `test_parser_handles_valid_json` — Clean LLM output parses
- `test_parser_handles_markdown_fences` — Dirty LLM output still parses
- `test_health` — `/health` returns 200
- `test_upload_rejects_bad_mime` — Bad file types get 400
- `test_upload_saves_to_local_disk` — File exists in UPLOAD_DIR after upload
- `test_delete_removes_audio_file` — File is cleaned up on session delete

---

## Summary

> [!TIP]
> **All free-tier services used throughout:** Groq API (LLM), HuggingFace (pyannote model access), faster-whisper (local, MIT licence). Audio is stored on the **local filesystem** — zero cloud storage cost. The entire backend can run in development at **$0 cost**.

### Data Flow — End to End

```
Upload → Local Disk → BullMQ → FFmpeg → pydub → [SepFormer] → faster-whisper → pyannote → DB
                                                                                    ↓
                                                          BullMQ → Groq LLM → Summary → API/Email
```

### What Was Removed

| Removed Feature | Reason | Replacement |
|---|---|---|
| **WebSocket live streaming** | Simplify Phase 1 scope — upload-only ingestion | File upload via multipart POST |
| **Cloudflare R2 / AWS S3** | No cloud dependency — reduce cost and complexity | Local filesystem (`UPLOAD_DIR`) |

> [!NOTE]
> Both removed features can be reintroduced later as incremental upgrades. WebSocket streaming adds a second ingestion path, and swapping local storage for S3/R2 only requires changing `app/services/storage.py`.
