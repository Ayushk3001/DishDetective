# Image → Recipe + YouTube (Streamlit + Autogen)

Two collaborating agents:
1) **Recipe_Generator**: looks at your uploaded food photo, identifies the dish, and prints a clean recipe (first line `DISH: <name>`).
2) **YT_Searcher**: calls a YouTube search **tool** and returns top links as a **table** (Title, URL, Channel, Duration, Views).

Built with **Autogen AgentChat**, **Streamlit**, and a lightweight YouTube search wrapper.

---

## Demo

- Upload a food image (JPG/PNG/WebP)
- Click **Analyze**
- See the identified dish + recipe and a table of YouTube videos

---

## Project Structure

```
.
├── app.py                 # Streamlit UI
├── backend.py             # Agents, tool, team, and orchestrator
├── requirements.txt       # Python dependencies
└── README.md
```

---

## How It Works

- `backend.py` defines:
  - **YouTube tool**: `youtube_search(query, max_results)` via `youtube-search`.
  - **Recipe_Generator**: analyzes the image and writes the recipe (first line `DISH: ...`).
  - **YT_Searcher**: reads the dish name, calls the tool, and returns results as a table.
  - **Orchestrator**: `run_orchestrator_with_bytes(image_bytes)` runs both agents and **extracts**:
    - Recipe text (from the Recipe_Generator message)
    - YouTube links (from tool events or a Markdown table, with robust fallbacks)

- `app.py` (Streamlit) calls the orchestrator and renders:
  - The **recipe** as Markdown
  - The **YouTube results** as a **table** (clickable links)

---

## Quickstart

**Requirements**
- Python **3.10 – 3.12** (tested on 3.11)
- An API key exported as `GEMINI_API_KEY` (used by `OpenAIChatCompletionClient`)

**1) Clone & create a virtual env**
```bash
git clone https://github.com/Ayushk3001/Image-to-Recipe-YouTube.git
cd Image-to-Recipe-YouTube
python -m venv .venv
# Windows
.\.venv\Scriptsctivate
# macOS/Linux
source .venv/bin/activate
```

**2) Install deps**
```bash
pip install -r requirements.txt
```

**3) Create `.env`**
```
GEMINI_API_KEY=YOUR_KEY_HERE
```

**4) Run the app**
```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), upload an image, and click **Analyze**.

---

## requirements.txt

```
youtube-search
autogen-agentchat
autogen-core
autogen-ext
asyncio
python-dotenv
openai
tiktoken
streamlit
pillow
requests
pandas
```

> Note: the import in code is `from dotenv import load_dotenv`, which corresponds to **python-dotenv** on PyPI.

---

## Configuration & Customization

- **Change the model**
  ```python
  # backend.py
  model_client = OpenAIChatCompletionClient(model="gemini-2.5-flash", api_key=api_key)
  ```
  Use any model supported by your `autogen-ext` version.

- **Adjust conversation depth**
  ```python
  team = RoundRobinGroupChat(participants=[Recipe_Generator, YT_Searcher], max_turns=4)
  ```

- **Control YouTube results**
  - Edit `max_results` inside `youtube_search`
  - Or tweak the YT agent prompt (e.g., add cuisine keywords)

---

## Robustness

The orchestrator avoids relying on “last message wins.” It:
- Pulls the recipe from the **Recipe_Generator** `TextMessage` (or any message with `DISH:` as fallback).
- Extracts YouTube data from **tool events** (`ToolCallExecutionEvent` / `ToolCallSummaryMessage`).
- If the model prints a **Markdown table**, it parses that.
- If the tool payload appears as a **raw list-of-dicts**, it parses that too.

---

## Acknowledgements

- [Autogen](https://github.com/microsoft/autogen) AgentChat  
- [Streamlit](https://streamlit.io/)  
- [`youtube-search`](https://pypi.org/project/youtube-search/)
