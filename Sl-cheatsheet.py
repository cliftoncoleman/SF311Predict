import streamlit as st

st.set_page_config(page_title="Streamlit Built-ins Cheat Sheet", page_icon="📘")

# === Titles & Headers ===
st.title("st.title → Big title (H1)")
st.header("st.header → Section header (H2)")
st.subheader("st.subheader → Sub-section header (H3-ish)")
st.caption("st.caption → Small gray caption text")

# === Basic Text ===
st.text("st.text → Plain fixed-width text")
st.write("st.write → Flexible text (supports strings, numbers, DataFrames, dicts, etc.)")
st.markdown("---")  # horizontal line

# === Status / Callouts ===
st.success("st.success → ✅ Success message")
st.info("st.info → ℹ️ Informational message")
st.warning("st.warning → ⚠️ Warning message")
st.error("st.error → ❌ Error message")

# === Code & Markdown ===
st.code("print('Hello, Streamlit!')", language="python")
st.markdown("Inline code: `df.head()` and **bold** or *italic* text.")

# === Layout Helpers ===
st.divider()  # horizontal rule
col1, col2 = st.columns(2)
with col1:
    st.subheader("Left Column")
    st.write("Put charts, text, anything here.")
with col2:
    st.subheader("Right Column")
    st.write("Side-by-side layout.")

# === Miscellaneous ===
st.metric("st.metric → Metric box", "123", "+5")
st.caption("Tip: use st.metric for KPIs or summary stats")

st.markdown("✅ That’s the core set of **Streamlit built-ins** you’ll use most often.")
