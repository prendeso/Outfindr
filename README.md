# Outfindr

Reddit outfit-ID bot. Summon `u/outfindr` on a post with an image; it identifies the
clothing items and replies with a structured breakdown plus shopping search links.

## Architecture

```
src/outfindr/
├── core/        # Platform-agnostic. Imports zero platform SDKs.
├── adapters/    # One file per platform. MVP ships only reddit_bot.
├── prompts/     # Static prompt text, versioned in filenames.
├── cli.py       # Local image/URL → JSON, for evals
└── worker.py    # Entrypoint: instantiates the Reddit adapter
```

The core/adapters split is the reusability invariant: adding a new platform
(Bluesky, Discord, web upload) means a new file in `adapters/` and zero
changes to `core/`.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,web]"

cp .env.example .env  # fill in keys
pytest
```

## Running the web eval UI

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export DATABASE_PATH=./outfindr-dev.db
uvicorn outfindr.web.app:app --reload --port 8000
```

Open http://localhost:8000. Upload a local image or paste an image URL to get
a structured outfit breakdown plus shopping search links. Repeat submissions
of the same image hit the SQLite cache and skip the vision call. Visit
`/history` for recent runs.

## Running the CLI eval

```bash
python -m outfindr.cli path/to/image.jpg
python -m outfindr.cli https://i.redd.it/xxx.jpg

# narrow with a free-text question
python -m outfindr.cli https://i.redd.it/xxx.jpg --query "the yellow jacket"
python -m outfindr.cli photo.jpg --query "second person from the left"
```

## Running the bot locally

```bash
python -m outfindr.worker
```

## Deployment

Railway worker. See `railway.toml` and `Procfile`. Volume mounted at `/data`
holds the SQLite cache + reply log.
