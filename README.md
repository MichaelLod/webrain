# WeBrain

**AI from the people, for the people.**

WeBrain is an open-source platform where anyone with a browser can contribute GPU compute to train a shared language model from scratch. No data centers. No corporations. Just people pooling their browsers to build AI that belongs to everyone.

## How it works

1. **Sign up** and get 100 free tokens
2. **Contribute** — your browser picks up tiny 64x64 matrix tiles and computes them on your GPU via WebGPU
3. **Earn & chat** — earn tokens for every tile, spend them to talk to the model as it gets smarter

The server decomposes every matrix multiplication in the training loop into small tile operations and distributes them to connected browsers. Each browser computes a single tile matmul and sends the result back. The server assembles everything and runs the training step. The model never needs to fit in a single GPU — it scales with the number of contributors.

## Architecture

```
Browser Workers (many)              Server
┌──────────────────┐        ┌──────────────────────────────┐
│  WebGPU Worker   │   WS   │  Training Orchestrator        │
│  ┌────────────┐  │◄──────►│  - Tile decomposition         │
│  │ tile matmul│  │        │  - Tile assembly               │
│  │ [64x64]    │  │        │  - Model weights (PyTorch)     │
│  └────────────┘  │        │  - Optimizer, loss, activations│
│  Stateless.      │        │  - Inference for chat          │
└──────────────────┘        └──────────────────────────────┘
```

- **Browsers** only compute tiny matrix tiles. Stateless. No model knowledge.
- **Server** handles everything else: weights, tiling, activations, loss, gradients, optimizer, data loading.

## Tech stack

| Layer | Stack |
|-------|-------|
| Frontend | Next.js, TypeScript, Tailwind, shadcn/ui |
| Backend | Python, FastAPI, PyTorch, SQLAlchemy |
| Database | PostgreSQL 16 |
| Compute | WebGPU (WGSL shaders) in Web Workers |
| Communication | REST + WebSocket (tiles) + SSE (chat) |

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

## The model

Tiny GPT-2 style transformer (~0.9M params):
- 4 transformer layers, 128 hidden dim, 4 attention heads
- Character-level tokenizer (256 vocab)
- The tiled architecture means this scales to any model size as the community grows

## Token economy

| Action | Tokens |
|--------|--------|
| Sign up bonus | +100 |
| Per tile computed | +1 (min), scales with compute time |
| Per chat message | -10 |

## Contributing

This is a community project. Contributions of all kinds are welcome:

- **Code** — features, bug fixes, performance improvements
- **Training data** — curate and contribute text corpora
- **Testing** — try it out, report bugs, suggest improvements
- **Ideas** — open an issue, start a discussion

### Development setup

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-thing`)
3. Make your changes
4. Run the frontend build (`cd frontend && npm run build`)
5. Open a PR

See [open issues](../../issues) for things to work on.

## Project structure

```
webrain/
├── frontend/              # Next.js app
│   └── src/
│       ├── app/           # Pages (landing, auth, dashboard, compute, chat)
│       ├── components/    # UI components
│       ├── workers/       # WebGPU compute (WGSL shader, GPU engine, worker)
│       ├── hooks/         # React hooks (auth, compute worker)
│       └── lib/           # API client, WebSocket client
├── backend/               # FastAPI app
│   └── app/
│       ├── api/v1/        # Routes (auth, tokens, compute, chat)
│       ├── models/        # SQLAlchemy models
│       ├── services/      # Token service, compute manager
│       ├── core/          # Config, database, security
│       └── ml/            # Model, tiling engine, trainer, inference
└── docker-compose.yml     # PostgreSQL
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Use it, fork it, build on it.
