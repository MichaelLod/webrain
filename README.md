# WeBrain

**The AI that lives in browsers.**

WeBrain is an open-source platform that distributes a language model across browser peers. No data centers, no corporations — just people pooling their GPUs through WebGPU to train, store, and run AI that belongs to everyone.

## How it works

1. **Sign up** and get 100 free tokens
2. **Contribute** — your browser loads transformer layers and computes forward passes via WebGPU
3. **Earn & chat** — earn tokens for every computation, spend them to talk to the model as it grows

The model lives across the network. Each browser stores and computes a subset of transformer layers, with activations flowing between peers. More browsers = more compute, more storage, smarter model.

## Architecture

```
                          WebRTC (direct P2P)
                    ┌─────────────────────────────┐
                    │                               │
                    ▼                               ▼
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│  Server  │───▶│ Browser A│───▶│ Browser B│───▶│ Browser C│───▶│  Server │
│  embed   │    │ layers   │    │ layers   │    │ layers   │    │  head   │
│          │    │  0 – 3   │    │  4 – 7   │    │  8 – 11  │    │         │
└─────────┘    └──────────┘    └──────────┘    └──────────┘    └─────────┘
    │               │               │               │               │
    └───────────────┴───────────────┴───────────────┴───────────────┘
                        WebSocket fallback relay
```

### Scaling modes

| Browsers | Mode | How it works |
|----------|------|-------------|
| 0 | Server Only | Full model runs on the server |
| 1 | Swarm | FFN split into 4 expert slices; browser computes 3, server computes 1 |
| 2–3 | Pipeline | Each browser handles a range of transformer layers, activations relayed via server |
| 4+ | Full P2P | Pipeline with WebRTC — activations flow browser-to-browser, bypassing the server |

### Key features

- **Pipeline parallelism** — transformer layers distributed across browser peers
- **WebRTC P2P** — direct browser-to-browser activation transfer for 4+ peers
- **IndexedDB persistence** — weights survive tab close, no re-download on return
- **Shard replication** — each layer stored on 2+ browsers for fault tolerance
- **Distributed training** — forward/backward passes flow through the pipeline
- **fp16 transfer** — half-precision weight downloads, converted to fp32 for WebGPU
- **Automatic fallback** — pipeline hops fall back to server relay if WebRTC fails

## The model

WeBrainGPT — 45M parameter multimodal transformer:

| Spec | Value |
|------|-------|
| Layers | 12 |
| Hidden dim | 512 |
| FFN dim | 1376 (SwiGLU) |
| Attention | GQA (8 heads, 4 KV heads) |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Tokenizer | BPE (32K vocab) |
| Max context | 512 tokens |
| Vision | 2-layer patch encoder |

## Tech stack

| Layer | Stack |
|-------|-------|
| Frontend | Next.js, TypeScript, Tailwind, shadcn/ui |
| Backend | Python, FastAPI, PyTorch, SQLAlchemy |
| Database | PostgreSQL 16 |
| Compute | WebGPU (WGSL shaders) in Web Workers |
| P2P | WebRTC DataChannels + WebSocket signaling |
| Storage | IndexedDB (browser weights), sharded checkpoints (server) |
| Communication | REST + WebSocket + SSE (chat) + WebRTC (activations) |

## Getting started

### Prerequisites

- Docker (for PostgreSQL)
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- Node.js 20+

### Run it

```bash
# Start the database
docker compose up -d

# Start the backend
cd backend
cp .env.example .env
uv sync
uv run uvicorn app.main:app --reload

# Start the frontend (new terminal)
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000 — register, contribute compute, and chat.

## Token economy

| Action | Tokens |
|--------|--------|
| Sign up bonus | +100 |
| Per task computed | +1 (min), scales with compute time |
| Per chat message | -10 |

## Project structure

```
webrain/
├── frontend/                  # Next.js app
│   └── src/
│       ├── app/               # Pages (landing, auth, dashboard, compute, chat, data)
│       ├── components/        # UI components (shadcn/ui)
│       ├── workers/           # WebGPU compute
│       │   ├── gpu-engine.ts          # WebGPU matmul + FFN engine
│       │   ├── compute-worker.ts      # Web Worker message handler
│       │   ├── pipeline-engine.ts     # Full transformer layer computation
│       │   ├── training-engine.ts     # Forward/backward for distributed training
│       │   └── shaders/               # WGSL shaders (matmul, rmsnorm, rope, softmax)
│       ├── hooks/             # React hooks (auth, compute worker)
│       └── lib/               # API client, WebSocket, WebRTC, P2P protocol, weight store
├── backend/                   # FastAPI app
│   └── app/
│       ├── api/v1/            # Routes (auth, tokens, compute, chat, data, weights)
│       ├── models/            # SQLAlchemy models
│       ├── services/          # Compute manager, shard registry, pipeline scheduler,
│       │                      # signaling, replication, weight server
│       ├── core/              # Config, database, security
│       └── ml/                # Model, trainer, inference, pipeline inference,
│                              # distributed trainer, sharded checkpoints, swarm
└── docker-compose.yml         # PostgreSQL
```

## Contributing

This is a community project. Contributions welcome:

- **Code** — features, bug fixes, performance
- **Training data** — submit URLs through the Data page
- **Testing** — try it out, report bugs
- **Ideas** — open an issue

### Development

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-thing`)
3. Make your changes
4. Run the frontend build (`cd frontend && npm run build`)
5. Open a PR

## License

Apache 2.0 — see [LICENSE](LICENSE).
