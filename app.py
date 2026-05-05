import html
import re
from typing import Dict, List

import streamlit as st

from backend import run_orchestrator_with_bytes


st.set_page_config(page_title="DishDetective", layout="wide")


CSS = """
<style>
:root {
    --ink: #22201c;
    --muted: #726c63;
    --line: #e3ddd1;
    --paper: #fffdf8;
    --surface: #f8f2e8;
    --tomato: #c84d34;
    --tomato-dark: #9f3a28;
    --sage: #4f7d66;
    --gold: #b9832f;
    --teal: #2f6d73;
}

[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(180deg, rgba(255, 253, 248, 0.98) 0%, rgba(248, 242, 232, 0.96) 42%, rgba(238, 244, 241, 0.92) 100%);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, li, label, span {
    letter-spacing: 0;
}

.hero {
    border: 1px solid rgba(34, 32, 28, 0.1);
    border-radius: 8px;
    background: linear-gradient(135deg, rgba(255, 253, 248, 0.96), rgba(250, 239, 221, 0.86));
    padding: 2rem;
    box-shadow: 0 18px 45px rgba(65, 48, 28, 0.08);
    margin-bottom: 1.4rem;
}

.eyebrow {
    color: var(--tomato-dark);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
}

.hero h1 {
    color: var(--ink);
    font-size: 3rem;
    line-height: 1.02;
    margin: 0 0 0.75rem 0;
}

.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    line-height: 1.65;
    max-width: 720px;
    margin: 0;
}

.step-row {
    display: grid;
    gap: 0.7rem;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin-top: 1.5rem;
}

.step {
    min-height: 92px;
    border: 1px solid rgba(34, 32, 28, 0.1);
    border-radius: 8px;
    background: rgba(255, 253, 248, 0.74);
    padding: 0.95rem;
}

.step strong {
    color: var(--ink);
    display: block;
    font-size: 0.94rem;
    margin-bottom: 0.35rem;
}

.step span {
    color: var(--muted);
    display: block;
    font-size: 0.86rem;
    line-height: 1.45;
}

.panel-title {
    color: var(--ink);
    font-size: 1.15rem;
    font-weight: 800;
    margin: 0 0 0.25rem 0;
}

.panel-copy {
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 0 0 0.8rem 0;
}

.placeholder {
    border: 1px dashed rgba(34, 32, 28, 0.24);
    border-radius: 8px;
    background: rgba(255, 253, 248, 0.62);
    min-height: 285px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: var(--muted);
    padding: 1.4rem;
}

.hint-strip {
    border-left: 4px solid var(--sage);
    background: rgba(79, 125, 102, 0.1);
    border-radius: 8px;
    color: #395f4d;
    font-size: 0.92rem;
    line-height: 1.55;
    padding: 0.95rem 1rem;
    margin-top: 1rem;
}

.dish-banner {
    border: 1px solid rgba(34, 32, 28, 0.1);
    border-radius: 8px;
    background: var(--paper);
    padding: 1.1rem 1.2rem;
    margin: 1.6rem 0 1rem 0;
}

.dish-label {
    color: var(--tomato-dark);
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

.dish-name {
    color: var(--ink);
    font-size: 1.9rem;
    font-weight: 850;
    line-height: 1.2;
    overflow-wrap: anywhere;
}

.result-title {
    color: var(--ink);
    font-size: 1.35rem;
    font-weight: 850;
    margin: 0.7rem 0 0.5rem 0;
}

.video-card {
    align-items: flex-start;
    border: 1px solid rgba(34, 32, 28, 0.1);
    border-radius: 8px;
    background: rgba(255, 253, 248, 0.9);
    display: flex;
    gap: 0.95rem;
    padding: 1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 12px 28px rgba(65, 48, 28, 0.06);
}

.video-index {
    align-items: center;
    background: #fff1e8;
    border: 1px solid rgba(200, 77, 52, 0.22);
    border-radius: 8px;
    color: var(--tomato-dark);
    display: flex;
    flex: 0 0 42px;
    font-size: 0.86rem;
    font-weight: 850;
    height: 42px;
    justify-content: center;
}

.video-title {
    color: var(--ink);
    font-size: 1rem;
    font-weight: 800;
    line-height: 1.35;
    margin-bottom: 0.45rem;
    overflow-wrap: anywhere;
}

.video-meta {
    color: var(--muted);
    display: flex;
    flex-wrap: wrap;
    font-size: 0.84rem;
    gap: 0.4rem 0.65rem;
    margin-bottom: 0.7rem;
}

.watch-link {
    align-items: center;
    background: var(--ink);
    border-radius: 8px;
    color: #fffdf8 !important;
    display: inline-flex;
    font-size: 0.86rem;
    font-weight: 800;
    min-height: 36px;
    padding: 0.45rem 0.8rem;
    text-decoration: none !important;
}

.empty-state {
    border: 1px solid rgba(34, 32, 28, 0.1);
    border-radius: 8px;
    background: rgba(255, 253, 248, 0.76);
    color: var(--muted);
    padding: 1.2rem;
}

div[data-testid="stFileUploader"] section {
    border: 1px dashed rgba(34, 32, 28, 0.24);
    border-radius: 8px;
    background: rgba(255, 253, 248, 0.72);
}

div[data-testid="stFileUploader"] button,
div[data-testid="stButton"] button {
    border-radius: 8px;
    font-weight: 800;
    min-height: 2.8rem;
}

div[data-testid="stButton"] button[kind="primary"] {
    background: var(--tomato);
    border-color: var(--tomato);
}

div[data-testid="stButton"] button[kind="primary"]:hover {
    background: var(--tomato-dark);
    border-color: var(--tomato-dark);
}

[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid rgba(34, 32, 28, 0.1);
    box-shadow: 0 14px 32px rgba(65, 48, 28, 0.1);
}

@media (max-width: 760px) {
    .block-container {
        padding-top: 1.1rem;
    }

    .hero {
        padding: 1.25rem;
    }

    .hero h1 {
        font-size: 2.1rem;
    }

    .step-row {
        grid-template-columns: 1fr;
    }

    .video-card {
        flex-direction: column;
    }
}
</style>
"""


def extract_dish_name(recipe_text: str) -> str:
    match = re.search(r"(?im)^DISH:\s*(.+)$", recipe_text or "")
    return match.group(1).strip() if match else "Detected dish"


def clean_recipe(recipe_text: str) -> str:
    text = re.sub(r"(?im)^DISH:\s*.+\n?", "", recipe_text or "").strip()
    return text or "The recipe response was empty. Try analyzing the image again."


def extract_url(value: str) -> str:
    value = value or ""
    markdown_match = re.search(r"\((https?://[^)]+)\)", value)
    if markdown_match:
        return markdown_match.group(1)
    plain_match = re.search(r"https?://[^\s|)]+", value)
    return plain_match.group(0).rstrip(".,") if plain_match else ""


def escape(value: str) -> str:
    return html.escape(str(value or ""), quote=True)


def render_video_card(row: Dict[str, str], index: int) -> None:
    title = escape(row.get("Title", "Untitled video"))
    channel = escape(row.get("Channel", "Unknown channel"))
    duration = escape(row.get("Duration", ""))
    views = escape(row.get("Views", ""))
    url = escape(extract_url(row.get("URL", "")))

    meta_items = [channel]
    if duration:
        meta_items.append(duration)
    if views:
        meta_items.append(views)
    meta_html = "".join(f"<span>{item}</span>" for item in meta_items if item)
    link_html = (
        f'<a class="watch-link" href="{url}" target="_blank" rel="noopener noreferrer">Watch video</a>'
        if url
        else '<span class="watch-link">No link found</span>'
    )

    st.markdown(
        f"""
        <article class="video-card">
            <div class="video-index">{index:02d}</div>
            <div>
                <div class="video-title">{title}</div>
                <div class="video-meta">{meta_html}</div>
                {link_html}
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def reset_results_if_new_upload(upload_key: str) -> None:
    if st.session_state.get("upload_key") != upload_key:
        st.session_state.upload_key = upload_key
        st.session_state.recipe_text = ""
        st.session_state.youtube_rows = []
        st.session_state.analysis_error = ""


st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">AI-powered kitchen assistant</div>
        <h1>DishDetective</h1>
        <p>
            Upload a food photo and get a clean dish identification, a practical recipe,
            and matching YouTube guides in one focused workspace.
        </p>
        <div class="step-row">
            <div class="step"><strong>1. Upload</strong><span>Add a clear JPG, PNG, or WebP food photo.</span></div>
            <div class="step"><strong>2. Analyze</strong><span>The agent identifies the dish and builds a recipe.</span></div>
            <div class="step"><strong>3. Cook</strong><span>Compare the recipe with video walkthroughs.</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.markdown('<div class="panel-title">Food Photo</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-copy">Choose the most appetizing, well-lit angle you have.</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload a food photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        upload_key = f"{uploaded.name}:{len(image_bytes)}"
        reset_results_if_new_upload(upload_key)
        st.image(image_bytes, caption=uploaded.name, use_container_width=True)
    else:
        image_bytes = b""
        st.markdown(
            '<div class="placeholder">Your uploaded food image will appear here.</div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="panel-title">Analysis Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-copy">Run the two-agent pipeline when your image is ready.</p>',
        unsafe_allow_html=True,
    )

    analyze_disabled = uploaded is None
    analyze_clicked = st.button(
        "Analyze dish",
        type="primary",
        use_container_width=True,
        disabled=analyze_disabled,
    )

    st.markdown(
        """
        <div class="hint-strip">
            Strong results usually come from images where the main dish fills the frame
            and sauces, toppings, or grains are visible.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analyze_clicked and image_bytes:
        with st.spinner("Identifying the dish, drafting the recipe, and finding video guides..."):
            try:
                recipe_text, youtube_rows = run_orchestrator_with_bytes(image_bytes)
                st.session_state.recipe_text = recipe_text
                st.session_state.youtube_rows = youtube_rows
                st.session_state.analysis_error = ""
            except Exception as exc:
                st.session_state.recipe_text = ""
                st.session_state.youtube_rows = []
                st.session_state.analysis_error = f"Unexpected error: {exc}"

    if st.session_state.get("analysis_error"):
        st.error(st.session_state.analysis_error)
    elif not st.session_state.get("recipe_text"):
        st.markdown(
            '<div class="empty-state">Upload a dish photo and run the analysis to see the recipe workspace.</div>',
            unsafe_allow_html=True,
        )


recipe_text = st.session_state.get("recipe_text", "")
youtube_rows: List[Dict[str, str]] = st.session_state.get("youtube_rows", [])

if recipe_text:
    dish_name = extract_dish_name(recipe_text)
    st.markdown(
        f"""
        <section class="dish-banner">
            <div class="dish-label">Detected dish</div>
            <div class="dish-name">{escape(dish_name)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    recipe_tab, video_tab = st.tabs(["Recipe", "Video guides"])

    with recipe_tab:
        st.markdown('<div class="result-title">Recipe</div>', unsafe_allow_html=True)
        st.markdown(clean_recipe(recipe_text))

    with video_tab:
        st.markdown('<div class="result-title">YouTube Guides</div>', unsafe_allow_html=True)
        if youtube_rows:
            for idx, row in enumerate(youtube_rows, start=1):
                render_video_card(row, idx)
        else:
            st.markdown(
                '<div class="empty-state">No YouTube results were parsed. Try another image or run the analysis again.</div>',
                unsafe_allow_html=True,
            )
