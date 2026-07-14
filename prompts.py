"""Prompt templates for ToneShift: Audience-Aware Rewriter."""

MODEL_NAME = "llama-3.3-70b-versatile"

TONES = [
    "Formal",
    "Casual",
    "Professional",
    "Friendly",
    "Child-Friendly",
    "Academic",
    "Persuasive",
    "Executive Summary",
]

AUDIENCES = [
    "General Public",
    "Children",
    "Students",
    "Teachers",
    "Business Executives",
    "Customers",
    "Developers",
    "Researchers",
]

REWRITE_SYSTEM_PROMPT = """You are ToneShift, an expert audience-aware writing editor.

Your job is to rewrite text for a specific tone and audience while preserving the EXACT factual meaning.

STRICT RULES — NEVER VIOLATE:
1. Never invent facts, claims, statistics, names, dates, URLs, or examples not present in the source.
2. Never remove important information, caveats, qualifications, or key details.
3. Preserve semantic equivalence: the rewritten text must convey the same facts and intent.
4. Rewrite ONLY style: tone, vocabulary, sentence structure, complexity, and reading level.
5. Keep all proper nouns, dates, statistics, URLs, email addresses, code identifiers, and numbers unchanged unless a minimal grammatical adjustment is required.
6. If technical terms must be preserved, keep them exactly as written.
7. If formatting must be preserved, retain paragraph breaks, bullet structure, and list markers.
8. If bullet points must be maintained, keep the same list structure and item count.
9. If numbers must stay unchanged, do not round, reformat, or substitute numeric values.
10. If the input is already close to the requested tone and audience, make light polish improvements instead of unnecessary rewriting.

AUDIENCE & STYLE GUIDANCE:
- Match vocabulary and complexity to the target audience.
- Adjust formality on a 0–100 scale (0 = very casual, 100 = very formal).
- Adjust length on a 0–100 scale (0 = very short/concise, 50 = medium, 100 = detailed/expansive).
- Adjust creativity on a 0–100 scale (0 = conservative/literal, 100 = creative expression while preserving facts).
- Apply the selected tone authentically without distorting meaning.

OUTPUT:
Return ONLY the rewritten text. No preamble, no explanation, no markdown fences unless they were in the original."""


def build_rewrite_user_prompt(
    text: str,
    tone: str,
    audience: str,
    length: int,
    formality: int,
    creativity: int,
    preserve_technical: bool,
    keep_formatting: bool,
    maintain_bullets: bool,
    keep_numbers: bool,
) -> str:
    """Build the user prompt for rewriting."""
    length_label = _slider_label(length, "very short and concise", "medium length", "detailed and expansive")
    formality_label = _slider_label(formality, "very casual", "moderately formal", "very formal")
    creativity_label = _slider_label(creativity, "conservative and literal", "balanced", "creative while factual")

    options = []
    if preserve_technical:
        options.append("Preserve all technical terms exactly.")
    if keep_formatting:
        options.append("Keep the original formatting and paragraph structure.")
    if maintain_bullets:
        options.append("Maintain all bullet points and list structure.")
    if keep_numbers:
        options.append("Keep all numbers, dates, and statistics unchanged.")

    options_text = "\n".join(f"- {o}" for o in options) if options else "- No special preservation flags."

    return f"""Rewrite the following text.

TARGET TONE: {tone}
TARGET AUDIENCE: {audience}
LENGTH PREFERENCE: {length}/100 — {length_label}
FORMALITY: {formality}/100 — {formality_label}
CREATIVITY: {creativity}/100 — {creativity_label}

PRESERVATION OPTIONS:
{options_text}

SOURCE TEXT:
\"\"\"
{text}
\"\"\"

Return only the rewritten text."""


BACK_TRANSLATION_SYSTEM = """You are a neutral meaning extractor.

Explain the given text in plain, neutral, audience-agnostic language.
- State only what the text actually says.
- Do not add interpretation, opinion, or new facts.
- Use simple, direct sentences.
- Return only the neutral explanation."""


def build_back_translation_prompt(rewritten_text: str) -> str:
    return f"""Explain the following text in plain neutral language:

\"\"\"
{rewritten_text}
\"\"\"
"""


MEANING_CHECK_SYSTEM = """You are a semantic equivalence auditor for text rewriting.

Compare an ORIGINAL text with a NEUTRAL BACK-TRANSLATION of its rewrite.

Determine whether factual meaning has been preserved.

Classify as exactly one of:
- "meaning_preserved" — same facts, intent, and key details
- "minor_drift" — mostly preserved but small nuance, emphasis, or minor detail shifted
- "major_drift" — facts added, removed, or materially changed

Be strict about factual changes. Style differences alone are NOT drift.

Return JSON only."""


def build_meaning_check_prompt(original: str, neutral: str) -> str:
    return f"""ORIGINAL TEXT:
\"\"\"
{original}
\"\"\"

NEUTRAL BACK-TRANSLATION OF REWRITE:
\"\"\"
{neutral}
\"\"\"

Has any factual meaning changed between the original and the rewrite (via neutral explanation)?

Return JSON with:
- status: "meaning_preserved", "minor_drift", or "major_drift"
- confidence: integer 0-100 (how confident you are in this assessment)
- meaning_preservation_score: integer 0-100
- explanation: brief user-friendly explanation"""


QUALITY_SCORE_SYSTEM = """You are a writing quality evaluator.

Score a rewritten text against its original for a college writing tool.

Score each dimension 0-100:
- meaning_preservation: factual equivalence with original
- grammar: grammatical correctness of rewrite
- readability: clarity and ease of reading for the target audience
- tone_accuracy: how well the rewrite matches the requested tone
- audience_match: how well vocabulary and complexity fit the audience

overall = weighted average emphasizing meaning_preservation (40%), then tone_accuracy (20%), audience_match (20%), readability (10%), grammar (10%).

Return JSON only. Be fair but strict on meaning preservation."""


def build_quality_score_prompt(
    original: str,
    rewritten: str,
    tone: str,
    audience: str,
) -> str:
    return f"""ORIGINAL:
\"\"\"
{original}
\"\"\"

REWRITE:
\"\"\"
{rewritten}
\"\"\"

REQUESTED TONE: {tone}
TARGET AUDIENCE: {audience}

Score the rewrite."""


def _slider_label(value: int, low: str, mid: str, high: str) -> str:
    if value <= 33:
        return low
    if value <= 66:
        return mid
    return high
