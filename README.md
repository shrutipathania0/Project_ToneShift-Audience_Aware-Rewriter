# ToneShift: Audience-Aware Rewriter

An AI-powered text rewriting application that adapts writing tone, vocabulary, sentence structure, complexity, and reading level for different audiences — while preserving the original meaning.

Built as a college mini-project using **Streamlit**, **Python**, and the **Groq API**.

---

## Project Overview

ToneShift is not a simple paraphraser. It intelligently rewrites text for a chosen tone and audience using carefully engineered prompts and Groq's `llama-3.3-70b-versatile` model. The app includes:

- Meaning drift detection via back-translation
- Side-by-side comparison with highlighted differences
- Multi-dimensional quality scoring
- Robust error handling

---

## Features

- **8 Tones:** Formal, Casual, Professional, Friendly, Child-Friendly, Academic, Persuasive, Executive Summary
- **8 Audiences:** General Public, Children, Students, Teachers, Business Executives, Customers, Developers, Researchers
- **Sliders:** Length, Formality, Creativity
- **Advanced Options:** Preserve technical terms, keep formatting, maintain bullets, keep numbers unchanged
- **Comparison View:** Word counts, reading time, similarity %, diff highlighting
- **Meaning Check:** Back-translation semantic audit with drift classification
- **Quality Scores:** Meaning, grammar, readability, tone accuracy, audience match
- **Safety Warning:** Alerts when meaning preservation confidence falls below 90%

---

## Requirements

- Python 3.10+
- Groq API key
- Internet connection

---

## Installation

1. Clone or download this project:

```bash
git clone <your-repo-url>
cd ToneShift
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

```bash
copy .env.example .env          # Windows
cp .env.example .env            # macOS/Linux
```

Edit `.env` and add your API key:

```env
GROQ_API_KEY=your_api_key_here
```

---

## How to Get a Groq API Key

1. Go to [Groq Console](https://console.groq.com/keys)
2. Sign in or create an account
3. Click **Create API Key**
4. Copy the key into your `.env` file as `GROQ_API_KEY`

Never commit your `.env` file or share your API key publicly.

---

## How to Run

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Project Structure

```
ToneShift/
├── app.py              # Streamlit UI
├── prompts.py          # Prompt engineering
├── utils.py            # Groq client, analysis, scoring
├── requirements.txt
├── .env.example
├── README.md
└── assets/
    └── style.css       # Custom UI styling
```

---

## Screenshots

| Rewrite Tab | Comparison Tab | Meaning Check |
|-------------|----------------|---------------|
| _Add screenshot here_ | _Add screenshot here_ | _Add screenshot here_ |

---

## Future Improvements

- Rewrite history and saved presets
- PDF/DOCX upload and export
- Custom user-defined tones
- Multi-language rewriting
- Team collaboration features
- Model selection dropdown (multiple Groq models)

---

## License

Educational use — college mini project.

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Groq API](https://console.groq.com/docs)
- [Groq Python SDK](https://github.com/groq/groq-python)
