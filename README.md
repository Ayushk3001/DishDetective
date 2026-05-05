# DishDetective

DishDetective turns a food photo into a dish name, a practical recipe, and matching YouTube recipe guides.

It uses:
- Streamlit for local development
- FastAPI for the Vercel deployment entrypoint
- AutoGen AgentChat for the two-agent workflow
- OpenAI API through `OpenAIChatCompletionClient`
- `youtube-search` for video guide lookup

## What It Does

1. Upload a food image in JPG, PNG, or WebP format.
2. Click **Analyze dish**.
3. The recipe agent identifies the dish and writes a clean recipe.
4. The YouTube agent searches for related recipe videos.
5. The app shows the recipe and video guides in separate tabs.

## Project Structure

```text
.
|-- app.py            # Streamlit UI for local development
|-- backend.py        # OpenAI client, AutoGen agents, YouTube tool, orchestrator
|-- index.py          # FastAPI ASGI app used by Vercel
|-- requirements.txt  # Python dependencies
|-- vercel.json       # Vercel function settings
|-- .env              # Local environment variables, not committed
`-- README.md
```

## How It Works

`backend.py` defines a two-agent pipeline:

- `Recipe_Generator` analyzes the uploaded image and returns the dish name plus recipe text. The first line is expected to be `DISH: <name>`.
- `YT_Searcher` reads the detected dish name, calls the `youtube_search` tool, and returns video results.
- `run_orchestrator_with_bytes(image_bytes)` runs the pipeline and extracts:
  - recipe text from the `Recipe_Generator` message
  - video rows from tool output or a Markdown table fallback

Each analysis run creates a fresh OpenAI model client, agents, and group chat team. This avoids reusing async event-loop state between Streamlit button clicks.

## Local Streamlit Quickstart

### Requirements

- Python 3.10 to 3.12
- An OpenAI API key

### 1. Create And Activate A Virtual Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env`

Create a `.env` file in this folder:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run The App

```bash
streamlit run app.py
```

Open the Streamlit URL, usually `http://localhost:8501`, upload a food image, and click **Analyze dish**.

## Deploy To Vercel

This repo includes `index.py`, a FastAPI app that exposes the required top-level ASGI variable:

```python
app = FastAPI(title="DishDetective")
```

Vercel requires a Python ASGI or WSGI app named `app` in an entrypoint such as `app.py`, `index.py`, or `api/index.py`. The Streamlit file `app.py` is kept for local use, while `index.py` is the deployment entrypoint.

Before deploying, add this environment variable in the Vercel project settings:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Then push to GitHub:

```bash
git add .
git commit -m "Add Vercel FastAPI entrypoint"
git push origin main
```

Vercel should detect the FastAPI dependency and deploy `index.py`.

## Configuration

### Change The OpenAI Model

In `backend.py`, update the model name inside `_build_team()`:

```python
model_client = OpenAIChatCompletionClient(model="gpt-5-nano", api_key=api_key)
```

Use another OpenAI model if your API account and installed AutoGen/OpenAI packages support it.

### Change The Number Of Video Results

Update the default value in `youtube_search`:

```python
def youtube_search(query: str, max_results: int = 5):
```

## Dependencies

Main packages used by the app:

- `streamlit`
- `fastapi`
- `python-multipart`
- `autogen-agentchat`
- `autogen-core`
- `autogen-ext`
- `openai`
- `python-dotenv`
- `youtube-search`
- `pillow`

## Troubleshooting

If the app says `OPENAI_API_KEY is missing`, make sure:

- `.env` exists in the same folder as `backend.py`
- the key name is exactly `OPENAI_API_KEY`
- Streamlit was restarted after creating or editing `.env`

If you change code while Streamlit is running, refresh the browser or restart Streamlit so the new backend code is loaded.

If Vercel says `No python entrypoint found`, make sure `index.py` is committed and contains the top-level FastAPI object named `app`.

## Acknowledgements

- [AutoGen](https://github.com/microsoft/autogen)
- [Streamlit](https://streamlit.io/)
- [`youtube-search`](https://pypi.org/project/youtube-search/)
