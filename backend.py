from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import Image as AGImage
from youtube_search import YoutubeSearch
from autogen_core.tools import FunctionTool
from PIL import Image
from io import BytesIO
import requests
from dotenv import load_dotenv
import os
import asyncio
from typing import List, Dict, Tuple, Any, Optional
import re, json, ast

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model_client = OpenAIChatCompletionClient(model="gpt-5-nano", api_key=api_key)

# ---------- YouTube tool ----------
def youtube_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    results = YoutubeSearch(query, max_results=max_results).to_dict()
    out: List[Dict[str, str]] = []
    for r in results:
        out.append(
            {
                "title": r.get("title", ""),
                "url": f"https://www.youtube.com{r.get('url_suffix', '')}",
                "channel": r.get("channel", ""),
                "duration": r.get("duration", ""),
                "views": r.get("views", ""),
            }
        )
    return out

youtube_search_tool = FunctionTool(
    youtube_search,
    name="youtube_search",
    description="Find relevant YouTube recipe videos for a given dish name or recipe query."
)

# ---------- Agents ----------
Recipe_Generator = AssistantAgent(
    name="Recipe_Generator",
    model_client=model_client,
    description=("Identifies the dish from an image and writes a clear recipe."),
    system_message=(
        "You analyze the provided food image.\n"
        "1) Output the dish name as the FIRST line exactly like: DISH: <dish name>\n"
        "2) Then give a concise, step-by-step recipe, with ingredients (with measures) and method.\n"
        "Keep the output clean and scannable."
    ),
)

YT_Searcher = AssistantAgent(
    name="YT_Searcher",
    model_client=model_client,
    description="Finds top YouTube links for the identified dish.",
    system_message=(
        "Wait for the previous message to contain the dish identified from the image. "
        "Read the line that starts with 'DISH:' to get the dish name. "
        "Then call the tool `youtube_search` with a query like '<dish name> recipe'. "
        "**Do NOT print raw JSON or Python lists.** "
        "Return the top 3-5 results as a Markdown table with headers exactly:\n"
        "| Title | URL | Channel | Duration | Views |\n"
        "Each row must contain those five columns. "
        "After the table, end your message with the word: DONE"
    ),
    tools=[youtube_search_tool]
)

# ---------- Team ----------
team = RoundRobinGroupChat(
    participants=[Recipe_Generator, YT_Searcher],
    max_turns=4
)

# ---------- Parsers / Extractors ----------
def _coerce_to_list_of_dicts(s: str) -> List[dict]:
    s = s.strip()
    # JSON first
    for cand in (s, s.replace("'", '"')):
        try:
            data = json.loads(cand)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except Exception:
            pass
    # Python literal
    try:
        data = ast.literal_eval(s)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except Exception:
        pass
    return []

def _normalize_rows(data: List[dict]) -> List[Dict[str, str]]:
    return [
        {
            "Title": d.get("title", ""),
            "URL": d.get("url", ""),
            "Channel": d.get("channel", ""),
            "Duration": d.get("duration", ""),
            "Views": d.get("views", ""),
        }
        for d in data
    ]

def _extract_tool_rows_from_messages(messages: List[Any]) -> List[Dict[str, str]]:
    """Pull list-of-dicts from ToolCallExecutionEvent / ToolCallSummaryMessage (from YT_Searcher)."""
    rows: List[Dict[str, str]] = []
    for m in reversed(messages):  # newest first
        if getattr(m, "source", "") != "YT_Searcher":
            continue
        mtype = getattr(m, "type", "")
        if mtype not in ("ToolCallExecutionEvent", "ToolCallSummaryMessage"):
            continue
        content = getattr(m, "content", None)
        # content may be [FunctionExecutionResult(...)] or a plain str
        if isinstance(content, list):
            for item in content:
                payload = getattr(item, "content", None)
                if isinstance(payload, str):
                    data = _coerce_to_list_of_dicts(payload)
                    if data:
                        return _normalize_rows(data)
        elif isinstance(content, str):
            data = _coerce_to_list_of_dicts(content)
            if data:
                return _normalize_rows(data)
    return rows

def _parse_markdown_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    lines = [ln.strip() for ln in text.splitlines()]
    start_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if ln.startswith("|") and ln.count("|") >= 6 and "Title" in ln and "URL" in ln:
            start_idx = i
            break
    if start_idx is None:
        return rows
    headers = [h.strip() for h in lines[start_idx].strip("|").split("|")]
    i = start_idx + 1
    # skip separator row if present
    if i < len(lines) and set(lines[i].replace("|", "").replace(" ", "")) <= set("-:"):
        i += 1
    while i < len(lines) and lines[i].startswith("|"):
        parts = [p.strip() for p in lines[i].strip("|").split("|")]
        if len(parts) == len(headers):
            row = dict(zip(headers, parts))
            rows.append({
                "Title": row.get("Title", ""),
                "URL": row.get("URL", ""),
                "Channel": row.get("Channel", ""),
                "Duration": row.get("Duration", ""),
                "Views": row.get("Views", ""),
            })
        i += 1
    return rows

# ---------- Orchestrator ----------
def run_orchestrator_with_bytes(image_bytes: bytes) -> Tuple[str, List[Dict[str, str]]]:

    async def _run(image_bytes_local: bytes) -> Tuple[str, List[Dict[str, str]]]:
        try:
            pil_image = Image.open(BytesIO(image_bytes_local)).convert("RGB")
        except Exception as e:
            return (f"Error reading image: {e}", [])

        ag_image = AGImage(pil_image)
        multi_modal = MultiModalMessage(
            content=[
                "Identify the dish and give me the recipe. Start with DISH: <name> on the first line. "
                "Then fetch YouTube links.",
                ag_image,
            ],
            source="user",
        )

        try:
            result = await team.run(task=multi_modal)
        except Exception as e:
            return (f"Agent run error: {e}", [])

        messages: List[Any] = getattr(result, "messages", []) or []

        # 1) Recipe: look for TextMessage from Recipe_Generator; fallback to any message containing DISH:
        recipe_text = ""
        for m in messages:
            if getattr(m, "type", "") == "TextMessage" and getattr(m, "source", "") == "Recipe_Generator":
                recipe_text = getattr(m, "content", "") or ""
                break
        if not recipe_text:
            for m in messages:
                txt = getattr(m, "content", "") or ""
                if "DISH:" in txt:
                    recipe_text = txt
                    break

        # 2) YouTube rows: prefer tool payloads; fallback to parsing text for table/list
        youtube_rows = _extract_tool_rows_from_messages(messages)

        # if no rows from tool events, try markdown table in the whole transcript
        if not youtube_rows:
            all_text = "\n\n".join([(getattr(m, "content", "") or str(m)) for m in messages])
            youtube_rows = _parse_markdown_table(all_text)
            if not youtube_rows:
                # last fallback: any list-of-dicts inside text
                data_blocks = _coerce_to_list_of_dicts(all_text)
                if data_blocks:
                    youtube_rows = _normalize_rows(data_blocks)

        recipe_text = (recipe_text or "").replace("DONE", "").strip()
        return (recipe_text, youtube_rows)

    return asyncio.run(_run(image_bytes))

