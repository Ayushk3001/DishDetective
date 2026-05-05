import base64
import html
import re
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from backend import run_orchestrator_with_bytes


app = FastAPI(title="DishDetective")


CSS = """
:root {
    --ink: #22201c;
    --muted: #726c63;
    --paper: #fffdf8;
    --surface: #f8f2e8;
    --tomato: #c84d34;
    --tomato-dark: #9f3a28;
    --sage: #4f7d66;
}

* {
    box-sizing: border-box;
}

body {
    background: linear-gradient(180deg, #fffdf8 0%, #f8f2e8 50%, #eef4f1 100%);
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0;
    min-height: 100vh;
}

.shell {
    margin: 0 auto;
    max-width: 1120px;
    padding: 34px 20px 54px;
}

.hero,
.panel,
.dish-banner,
.video-card,
.empty-state {
    border: 1px solid rgba(34, 32, 28, 0.1);
    border-radius: 8px;
    background: rgba(255, 253, 248, 0.88);
}

.hero {
    background: linear-gradient(135deg, rgba(255, 253, 248, 0.98), rgba(250, 239, 221, 0.9));
    box-shadow: 0 18px 45px rgba(65, 48, 28, 0.08);
    margin-bottom: 22px;
    padding: 30px;
}

.eyebrow {
    color: var(--tomato-dark);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0;
    margin-bottom: 10px;
    text-transform: uppercase;
}

h1 {
    font-size: clamp(42px, 7vw, 64px);
    letter-spacing: 0;
    line-height: 0.98;
    margin: 0 0 12px;
}

.hero p,
.panel-copy,
.meta,
.empty-state {
    color: var(--muted);
    line-height: 1.6;
}

.hero p {
    font-size: 17px;
    max-width: 760px;
    margin: 0;
}

.grid {
    display: grid;
    gap: 20px;
    grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
}

.panel {
    padding: 20px;
}

.panel-title {
    font-size: 19px;
    font-weight: 850;
    margin: 0 0 4px;
}

.panel-copy {
    font-size: 14px;
    margin: 0 0 14px;
}

form {
    display: grid;
    gap: 14px;
}

input[type="file"] {
    background: rgba(255, 253, 248, 0.8);
    border: 1px dashed rgba(34, 32, 28, 0.26);
    border-radius: 8px;
    color: var(--muted);
    min-height: 58px;
    padding: 16px;
    width: 100%;
}

button,
.link-button {
    align-items: center;
    background: var(--tomato);
    border: 1px solid var(--tomato);
    border-radius: 8px;
    color: #fffdf8;
    cursor: pointer;
    display: inline-flex;
    font-size: 15px;
    font-weight: 850;
    justify-content: center;
    min-height: 46px;
    padding: 10px 16px;
    text-decoration: none;
}

button:hover,
.link-button:hover {
    background: var(--tomato-dark);
    border-color: var(--tomato-dark);
}

.preview {
    border-radius: 8px;
    border: 1px solid rgba(34, 32, 28, 0.1);
    margin-top: 14px;
    max-width: 100%;
}

.dish-banner {
    margin: 22px 0;
    padding: 18px 20px;
}

.dish-label {
    color: var(--tomato-dark);
    font-size: 12px;
    font-weight: 850;
    text-transform: uppercase;
}

.dish-name {
    font-size: 30px;
    font-weight: 850;
    line-height: 1.2;
    overflow-wrap: anywhere;
}

.section-title {
    font-size: 22px;
    font-weight: 850;
    margin: 0 0 10px;
}

.recipe {
    color: var(--ink);
    font-size: 15px;
    line-height: 1.7;
    white-space: pre-wrap;
}

.video-card {
    display: flex;
    gap: 14px;
    margin-bottom: 12px;
    padding: 16px;
}

.video-index {
    align-items: center;
    background: #fff1e8;
    border: 1px solid rgba(200, 77, 52, 0.22);
    border-radius: 8px;
    color: var(--tomato-dark);
    display: flex;
    flex: 0 0 42px;
    font-weight: 850;
    height: 42px;
    justify-content: center;
}

.video-title {
    font-weight: 850;
    line-height: 1.35;
    margin-bottom: 6px;
    overflow-wrap: anywhere;
}

.meta {
    display: flex;
    flex-wrap: wrap;
    font-size: 13px;
    gap: 8px 12px;
    margin-bottom: 10px;
}

.watch-link {
    color: var(--sage);
    font-weight: 850;
}

.empty-state {
    padding: 16px;
}

@media (max-width: 780px) {
    .grid {
        grid-template-columns: 1fr;
    }

    .hero {
        padding: 22px;
    }

    .video-card {
        flex-direction: column;
    }
}
"""


def _page(content: str) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DishDetective</title>
    <style>{CSS}</style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <div class="eyebrow">AI-powered kitchen assistant</div>
            <h1>DishDetective</h1>
            <p>Upload a food photo and get a clean dish identification, a practical recipe, and matching YouTube guides.</p>
        </section>
        {content}
    </main>
</body>
</html>"""
    )


def _upload_panel(message: str = "") -> str:
    notice = f'<div class="empty-state">{html.escape(message)}</div>' if message else ""
    return f"""
<section class="grid">
    <div class="panel">
        <h2 class="panel-title">Food Photo</h2>
        <p class="panel-copy">Choose a clear JPG, PNG, or WebP image where the main dish fills the frame.</p>
        <form action="/analyze" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/jpeg,image/png,image/webp" required>
            <button type="submit">Analyze dish</button>
        </form>
    </div>
    <div class="panel">
        <h2 class="panel-title">Analysis Studio</h2>
        <p class="panel-copy">The OpenAI-powered agents identify the dish, draft the recipe, and search for useful video guides.</p>
        {notice or '<div class="empty-state">Upload an image to start the recipe workflow.</div>'}
    </div>
</section>
"""


def _extract_dish_name(recipe_text: str) -> str:
    match = re.search(r"(?im)^DISH:\s*(.+)$", recipe_text or "")
    return match.group(1).strip() if match else "Detected dish"


def _clean_recipe(recipe_text: str) -> str:
    return re.sub(r"(?im)^DISH:\s*.+\n?", "", recipe_text or "").strip()


def _extract_url(value: str) -> str:
    value = value or ""
    markdown_match = re.search(r"\((https?://[^)]+)\)", value)
    if markdown_match:
        return markdown_match.group(1)
    plain_match = re.search(r"https?://[^\s|)]+", value)
    return plain_match.group(0).rstrip(".,") if plain_match else ""


def _render_videos(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return '<div class="empty-state">No YouTube results were parsed. Try another image or run the analysis again.</div>'

    cards = []
    for index, row in enumerate(rows, start=1):
        title = html.escape(row.get("Title", "Untitled video"))
        channel = html.escape(row.get("Channel", "Unknown channel"))
        duration = html.escape(row.get("Duration", ""))
        views = html.escape(row.get("Views", ""))
        url = html.escape(_extract_url(row.get("URL", "")))
        meta_items = "".join(f"<span>{item}</span>" for item in (channel, duration, views) if item)
        link = (
            f'<a class="watch-link" href="{url}" target="_blank" rel="noopener noreferrer">Watch video</a>'
            if url
            else '<span class="meta">No link found</span>'
        )
        cards.append(
            f"""
<article class="video-card">
    <div class="video-index">{index:02d}</div>
    <div>
        <div class="video-title">{title}</div>
        <div class="meta">{meta_items}</div>
        {link}
    </div>
</article>"""
        )
    return "".join(cards)


def _result_page(image_bytes: bytes, content_type: str, recipe_text: str, youtube_rows: List[Dict[str, str]]) -> str:
    image_data = base64.b64encode(image_bytes).decode("ascii")
    image_src = f"data:{content_type};base64,{image_data}"
    dish_name = html.escape(_extract_dish_name(recipe_text))
    recipe = html.escape(_clean_recipe(recipe_text) or "The recipe response was empty. Try analyzing the image again.")

    return f"""
<a class="link-button" href="/">Analyze another image</a>
<section class="dish-banner">
    <div class="dish-label">Detected dish</div>
    <div class="dish-name">{dish_name}</div>
</section>
<section class="grid">
    <div class="panel">
        <h2 class="section-title">Uploaded Image</h2>
        <img class="preview" src="{image_src}" alt="Uploaded food image">
    </div>
    <div class="panel">
        <h2 class="section-title">Recipe</h2>
        <div class="recipe">{recipe}</div>
    </div>
</section>
<section class="panel" style="margin-top: 20px;">
    <h2 class="section-title">YouTube Guides</h2>
    {_render_videos(youtube_rows)}
</section>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return _page(_upload_panel())


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)) -> HTMLResponse:
    content_type = file.content_type or ""
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        return _page(_upload_panel("Please upload a JPG, PNG, or WebP food image."))

    image_bytes = await file.read()
    if not image_bytes:
        return _page(_upload_panel("The uploaded image was empty. Try another file."))

    recipe_text, youtube_rows = run_orchestrator_with_bytes(image_bytes)
    return _page(_result_page(image_bytes, content_type, recipe_text, youtube_rows))
