import streamlit as st
import pandas as pd  
from backend import run_orchestrator_with_bytes

st.set_page_config(page_title="DishDetective", page_icon="🍋")
st.title("🍋 Image → Recipe + YouTube Finder")

st.write(
    "Upload a food image. The first agent identifies the dish and writes a recipe. "
    "The second agent fetches related YouTube results and we show them in a table."
)

uploaded = st.file_uploader("Upload a food photo (JPG/PNG/WebP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    st.image(uploaded, caption="Your upload", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Running the two-agent pipeline..."):
            try:
                image_bytes = uploaded.read()
                recipe_text, youtube_rows = run_orchestrator_with_bytes(image_bytes)
            except Exception as e:
                recipe_text, youtube_rows = (f"Unexpected error: {e}", [])

        st.subheader("Recipe")
        # Prefer markdown so headings, lists, etc. render nicely
        st.markdown(recipe_text)

        st.subheader("YouTube Results")
        if youtube_rows:
            df = pd.DataFrame(youtube_rows, columns=["Title", "URL", "Channel", "Duration", "Views"])
            # Make URLs clickable in Streamlit tables by converting to markdown links
            df_display = df.copy()
            df_display["URL"] = df_display["URL"].apply(lambda u: f"[link]({u})" if u else "")
            st.table(df_display)
        else:
            st.info("No YouTube results parsed. Try another image or re-run.")
else:
    st.info("↑ Upload an image to begin.")

