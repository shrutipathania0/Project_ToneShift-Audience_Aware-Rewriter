"""ToneShift: Audience-Aware Rewriter — Streamlit application."""

from __future__ import annotations

import html
import json
import pathlib

import streamlit as st
import streamlit.components.v1 as components

from prompts import AUDIENCES, TONES
from utils import (
    ToneShiftError,
    check_meaning_drift,
    count_characters,
    count_words,
    create_client,
    estimate_reading_time_minutes,
    evaluate_quality,
    get_api_key,
    highlight_word_differences,
    rewrite_text,
    similarity_percentage,
    status_badge,
)

APP_DIR = pathlib.Path(__file__).parent
CSS_PATH = APP_DIR / "assets" / "style.css"

st.set_page_config(
    page_title="ToneShift: Audience-Aware Rewriter",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

if CSS_PATH.exists():
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap" rel="stylesheet">
        <style>{CSS_PATH.read_text(encoding='utf-8')}</style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "input_text": "",
        "output_text": "",
        "meaning_result": None,
        "quality_scores": None,
        "last_tone": TONES[2],
        "last_audience": AUDIENCES[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    st.markdown(
        """
        <div class="ts-hero">
            <h1>✨ ToneShift</h1>
            <p>Audience-Aware Rewriter — adapt tone, vocabulary &amp; complexity without changing meaning.</p>
            <div style="display:flex;gap:0.75rem;justify-content:center;flex-wrap:wrap;margin-top:1.2rem;">
                <div style="background:#fff3e0;border-radius:50px;padding:0.45rem 1.1rem;font-weight:800;font-size:0.85rem;color:#b85c00;box-shadow:0 4px 12px rgba(255,140,66,0.20);">
                    🎙️ 8 Tones
                </div>
                <div style="background:#fce4ec;border-radius:50px;padding:0.45rem 1.1rem;font-weight:800;font-size:0.85rem;color:#a0004a;box-shadow:0 4px 12px rgba(255,107,157,0.20);">
                    👥 8 Audiences
                </div>
                <div style="background:#e3f2fd;border-radius:50px;padding:0.45rem 1.1rem;font-weight:800;font-size:0.85rem;color:#0050a0;box-shadow:0 4px 12px rgba(91,184,245,0.20);">
                    🧠 Meaning Check
                </div>
                <div style="background:#e8f5e9;border-radius:50px;padding:0.45rem 1.1rem;font-weight:800;font-size:0.85rem;color:#1b5e20;box-shadow:0 4px 12px rgba(46,204,113,0.20);">
                    📊 Quality Score
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_counters(text: str) -> None:
    chars = count_characters(text)
    words = count_words(text)
    st.markdown(
        f'<p class="counter-text">Characters: <b>{chars}</b> &nbsp;|&nbsp; Words: <b>{words}</b></p>',
        unsafe_allow_html=True,
    )


def copy_button(label: str, text: str, key: str) -> None:
    if st.button(label, key=key, use_container_width=True):
        safe_text = json.dumps(text)
        components.html(
            f"""
            <script>
            navigator.clipboard.writeText({safe_text});
            </script>
            """,
            height=0,
        )
        st.toast("Copied to clipboard!", icon="📋")


def score_card(title: str, score: int) -> None:
    st.markdown(
        f"""
        <div class="ts-card">
            <h4>{title}</h4>
            <div class="score">{score}</div>
            <div class="ts-metric-label">out of 100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quality_scores(scores) -> None:
    st.subheader("📊 Quality Scores")
    cols = st.columns(5)
    metrics = [
        ("Meaning Preservation", scores.meaning_preservation),
        ("Grammar", scores.grammar),
        ("Readability", scores.readability),
        ("Tone Accuracy", scores.tone_accuracy),
        ("Audience Match", scores.audience_match),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            score_card(label, value)

    st.markdown(
        f"""
        <div class="ts-card">
            <h4>Overall Score</h4>
            <div class="score">{scores.overall}</div>
            <div class="ts-metric-label">weighted composite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rewrite_tab(api_key: str | None) -> None:
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("### 📝 Input Text")
        input_text = st.text_area(
            "input_text_area",
            value=st.session_state.input_text,
            height=260,
            placeholder="Paste or type your text here...",
            label_visibility="collapsed",
        )
        st.session_state.input_text = input_text
        render_counters(input_text)

        with st.container():
            st.markdown("#### 🎛️ Rewrite Controls")
            c1, c2 = st.columns(2)
            with c1:
                tone = st.selectbox("Tone", TONES, index=2)
            with c2:
                audience = st.selectbox("Target Audience", AUDIENCES)

            length = st.slider("Length", 0, 100, 50, help="0 = Very Short, 50 = Medium, 100 = Detailed")
            formality = st.slider("Formality", 0, 100, 50, help="0 = Very Casual, 100 = Very Formal")
            creativity = st.slider("Creativity", 0, 100, 35, help="0 = Conservative, 100 = Creative")

            st.markdown("##### ⚙️ Advanced Options")
            a1, a2 = st.columns(2)
            with a1:
                preserve_technical = st.checkbox("Preserve technical terms", value=True)
                keep_formatting = st.checkbox("Keep original formatting", value=True)
            with a2:
                maintain_bullets = st.checkbox("Maintain bullet points", value=True)
                keep_numbers = st.checkbox("Keep numbers unchanged", value=True)

            rewrite_clicked = st.button("🚀 Rewrite Text", use_container_width=True, type="primary")

    with right:
        st.markdown("### ✨ Rewritten Output")
        output_text = st.session_state.output_text

        if output_text:
            st.markdown(
                f'<div class="ts-output-box">{html.escape(output_text)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ts-output-box" style="color:#94a3b8;">Your rewritten text will appear here...</div>',
                unsafe_allow_html=True,
            )

        b1, b2, b3 = st.columns(3)
        with b1:
            copy_button("📋 Copy", output_text, "copy_output")
        with b2:
            st.download_button(
                "⬇️ Download",
                data=output_text or "",
                file_name="toneshift_rewrite.txt",
                mime="text/plain",
                use_container_width=True,
                disabled=not output_text,
            )
        with b3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.output_text = ""
                st.session_state.meaning_result = None
                st.session_state.quality_scores = None
                st.rerun()

        if st.session_state.quality_scores:
            render_quality_scores(st.session_state.quality_scores)

        meaning = st.session_state.meaning_result
        if meaning and meaning.meaning_preservation_score < 90:
            st.markdown(
                f"""
                <div class="ts-warning">
                    <b>⚠️ Meaning Preservation Warning</b><br>
                    Confidence is below 90% ({meaning.meaning_preservation_score}%). Review the rewrite carefully.
                    <br><br>{html.escape(meaning.explanation)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    if rewrite_clicked:
        if not api_key:
            st.warning("⚠️ Missing API key. Create a `.env` file with `GROQ_API_KEY=your_api_key`.")
            return
        if not input_text.strip():
            st.warning("⚠️ Please enter some text before rewriting.")
            return

        progress = st.progress(0, text="Preparing rewrite...")
        try:
            client = create_client(api_key)
            progress.progress(20, text="Rewriting with Groq...")
            result = rewrite_text(
                client,
                text=input_text,
                tone=tone,
                audience=audience,
                length=length,
                formality=formality,
                creativity=creativity,
                preserve_technical=preserve_technical,
                keep_formatting=keep_formatting,
                maintain_bullets=maintain_bullets,
                keep_numbers=keep_numbers,
            )
            st.session_state.output_text = result.text
            st.session_state.last_tone = tone
            st.session_state.last_audience = audience

            progress.progress(55, text="Evaluating quality...")
            st.session_state.quality_scores = evaluate_quality(
                client,
                original=input_text,
                rewritten=result.text,
                tone=tone,
                audience=audience,
            )

            progress.progress(80, text="Running meaning drift check...")
            st.session_state.meaning_result = check_meaning_drift(
                client,
                original=input_text,
                rewritten=result.text,
            )

            progress.progress(100, text="Done!")
            st.success("Rewrite completed successfully!")
            st.rerun()

        except ToneShiftError as exc:
            progress.empty()
            if exc.category == "rate_limit":
                st.warning(f"⚠️ {exc.message}")
            elif exc.category in {"network", "server"}:
                st.warning(f"⚠️ {exc.message}")
            elif exc.category == "auth":
                st.error(f"🔑 {exc.message}")
            else:
                st.error(f"❌ {exc.message}")
        except Exception as exc:
            progress.empty()
            st.error(f"❌ Unexpected error: {exc}")


def render_comparison_tab() -> None:
    original = st.session_state.input_text
    rewritten = st.session_state.output_text

    if not original or not rewritten:
        st.info("Generate a rewrite first to use the comparison view.")
        return

    st.subheader("🔍 Side-by-Side Comparison")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Original Words", count_words(original))
    with m2:
        st.metric("Rewritten Words", count_words(rewritten))
    with m3:
        st.metric("Similarity", f"{similarity_percentage(original, rewritten)}%")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Original Reading Time", f"{estimate_reading_time_minutes(original)} min")
    with c2:
        st.metric("Rewritten Reading Time", f"{estimate_reading_time_minutes(rewritten)} min")

    highlighted_orig, highlighted_rew = highlight_word_differences(original, rewritten)

    st.markdown("#### Original Text")
    st.markdown(f'<div class="ts-card">{highlighted_orig}</div>', unsafe_allow_html=True)
    st.markdown('<div class="ts-arrow">↓</div>', unsafe_allow_html=True)
    st.markdown("#### Rewritten Text")
    st.markdown(f'<div class="ts-card">{highlighted_rew}</div>', unsafe_allow_html=True)

    st.caption("Green = added/changed wording. Red strikethrough = removed/changed from original.")


def render_meaning_tab(api_key: str | None) -> None:
    original = st.session_state.input_text
    rewritten = st.session_state.output_text
    meaning = st.session_state.meaning_result

    if not original or not rewritten:
        st.info("Generate a rewrite first to run the meaning check.")
        return

    st.subheader("🧠 Meaning Drift Detector")
    st.markdown(
        """
        **Process:** Original → Rewrite → Neutral Back-Translation → Semantic Comparison
        """
    )

    if st.button("🔁 Re-run Meaning Check", use_container_width=False):
        if not api_key:
            st.warning("⚠️ Missing API key.")
            return
        try:
            with st.spinner("Analyzing meaning preservation..."):
                client = create_client(api_key)
                st.session_state.meaning_result = check_meaning_drift(
                    client,
                    original=original,
                    rewritten=rewritten,
                )
            st.success("Meaning check completed.")
            st.rerun()
        except ToneShiftError as exc:
            st.error(exc.message)

    if meaning:
        label, css_class = status_badge(meaning.status)
        st.markdown(f'<div class="ts-{css_class}"><b>{label}</b></div>', unsafe_allow_html=True)

        st.markdown("#### Neutral Back-Translation")
        st.markdown(
            f'<div class="ts-output-box">{html.escape(meaning.neutral_text)}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Explanation")
        st.write(meaning.explanation)

        st.metric("Meaning Preservation Score", f"{meaning.meaning_preservation_score}%")
        st.metric("Confidence", f"{meaning.confidence}%")

        if meaning.meaning_preservation_score < 90:
            st.warning(
                "Meaning preservation confidence is below 90%. Consider adjusting tone, length, or creativity."
            )


def render_about_tab() -> None:
    st.subheader("ℹ️ About ToneShift")
    st.markdown(
        """
        **ToneShift: Audience-Aware Rewriter** is a college mini-project that uses Groq
        to intelligently adapt writing style for different tones and audiences.

        Unlike simple paraphrasers, ToneShift changes:
        - Writing tone
        - Vocabulary
        - Sentence structure
        - Complexity and reading level

        while preserving the original meaning.

        ### 🔑 Core Features
        - 8 tone presets and 8 audience profiles
        - Length, formality, and creativity controls
        - Advanced preservation options
        - Comparison view with highlighted differences
        - Meaning drift detection via back-translation
        - Multi-dimensional quality scoring

        ### 🛠️ Tech Stack
        - **Frontend:** Streamlit
        - **Backend:** Python
        - **LLM:** Groq (`llama-3.3-70b-versatile`)
        - **Config:** python-dotenv

        ### 📸 Screenshots
        _Add screenshots of the Rewrite, Comparison, and Meaning Check tabs here._

        ### 🚀 Future Improvements
        - User accounts and rewrite history
        - Batch document processing
        - PDF/DOCX import and export
        - Custom tone presets
        - Multi-language support
        """
    )


def main() -> None:
    init_session_state()
    render_header()

    api_key = get_api_key()
    if not api_key:
        st.warning(
            "⚠️ **API key not found.**\n\n"
            "**Running locally?** Create a `.env` file in the project root:\n"
            "```\nGROQ_API_KEY=your_api_key\n```\n\n"
            "**Deployed on Streamlit Cloud?** Go to your app dashboard → "
            "**Settings → Secrets** and add:\n"
            "```toml\nGROQ_API_KEY = \"your_api_key\"\n```\n\n"
            "Get your free API key at [console.groq.com](https://console.groq.com) 🔑"
        )

    tab_rewrite, tab_compare, tab_meaning, tab_about = st.tabs(
        ["✍️ Rewrite", "🔍 Comparison", "🧠 Meaning Check", "ℹ️ About"]
    )

    with tab_rewrite:
        render_rewrite_tab(api_key)

    with tab_compare:
        render_comparison_tab()

    with tab_meaning:
        render_meaning_tab(api_key)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
