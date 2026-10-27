"""
SentryDoc AI - real-time multimodal document verification & fraud screening.

Run:  streamlit run app.py
Env:  GEMINI_API_KEY (or paste it in the Settings box), GEMINI_MODEL (optional)

NOTE: The authenticity score is an AI-generated risk signal for triage, not
forensic proof. Always route high-risk or borderline results to a human reviewer.
"""

from __future__ import annotations


import hashlib
import io
import json
import logging
import os
import time
from html import escape
from typing import Any

import pymupdf  # PyMuPDF - reads PDFs with no Poppler/PATH setup
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

try:  # optional: load variables from a local .env file
    from dotenv import load_dotenv

    load_dotenv(override=True)  # .env wins over stale system variables
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sentrydoc")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Gemini 2.5 models were shut down on 16 Oct 2026, so the app now asks the API
# which models your key can use and lets you pick one in Settings.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
PREFERRED_MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash"]
EXCLUDED_MODEL_WORDS = ("image", "tts", "live", "embedding", "audio", "computer", "robotics", "veo", "imagen")
MAX_FILE_MB = 15
MAX_PDF_PAGES = 5          # pages sent to the model (cost/latency guard)
PDF_DPI = 150
MAX_IMAGE_SIDE = 2000      # px
MAX_RETRIES = 3
ALLOWED_TYPES = ["pdf", "png", "jpg", "jpeg"]


class SentryDocError(Exception):
    """User-facing error with a message that explains how to fix the problem."""


def clean_key(raw: str | None) -> str:
    """Remove whitespace, newlines and stray quotes that break API keys."""
    return (raw or "").strip().strip("\"'").strip()


# --------------------------------------------------------------------------- #
# Response schema (enforced by Gemini via structured output)
# --------------------------------------------------------------------------- #
class FieldEntry(BaseModel):
    name: str = Field(description="Field label, e.g. 'Name', 'Issue Date'")
    value: str = Field(description="Value exactly as printed; 'Not found' if absent")


class DocumentAnalysis(BaseModel):
    document_type: str
    extracted_fields: list[FieldEntry]
    authenticity_score: int = Field(description="0 (certainly forged) to 100 (no concerns)")
    tampering_detected: bool
    flagged_reasons: list[str]
    summary: str = Field(description="Two-sentence plain-language verdict")


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #
ANALYSIS_SYSTEM_PROMPT = """\
You are SentryDoc, a document-forensics analyst who screens official documents \
(certificates, marksheets, transcripts, ID cards, letters) for fraud.

Work only from what is visible in the provided page images. Never invent values.

STEP 1 - Identify the document type and extract key fields (Name, Issue Date, \
ID/Roll/Certificate Number, Issuing Authority, plus other prominent fields). \
Copy values exactly as printed.

STEP 2 - Inspect for tampering and inconsistency:
  * Typography: mixed fonts, sizes, kerning or baselines within one field/line.
  * Alignment: text that drifts off its row/column, uneven spacing around values.
  * Pixel artifacts: halos, blur patches, noise or compression differences, \
cloned or pasted regions, erased-and-retyped areas, mismatched background texture.
  * Seals/stamps/signatures: missing where expected, blurry vs. surrounding \
content, duplicated, floating over text unnaturally, or low resolution.
  * Logical consistency: impossible/future dates, issue date before the \
period it certifies, grade/marks arithmetic that does not add up, ID numbers \
that break a visible format, misspelled institution names, mismatch between \
pages.
  * Template issues: missing watermarks/security features typical for this \
document type, unprofessional layout.

SCORING RULES
  * authenticity_score: 90-100 = no concerns; 70-89 = minor oddities, likely \
genuine; 40-69 = suspicious, needs human review; 0-39 = strong evidence of forgery.
  * Do NOT penalize ordinary phone-photo blur, skew, shadows, or scan noise by \
themselves; only flag issues that are localized or inconsistent.
  * tampering_detected is true only when you can point to specific visual or \
logical evidence. Each flagged_reasons entry must name WHAT is wrong and WHERE \
(page and area, e.g. "Page 1: 'Date of Birth' uses a different font weight \
than neighbouring fields"). Return an empty list if nothing is wrong.
  * If the image is too low quality to judge, say so in flagged_reasons and \
lower your confidence in the score instead of guessing.
  * You cannot verify the document against an issuer's database; say so in the \
summary when relevant.

Respond with JSON matching the provided schema and nothing else.
"""

ANALYSIS_USER_PROMPT = (
    "Analyze the attached document ({n} page image(s), in order). "
    "Return the structured verification report."
)

CHAT_SYSTEM_PROMPT = """\
You are SentryDoc's document assistant. The user uploaded the document shown in \
the images. Answer questions about that document only, grounded in what is \
visible (and the prior verification report provided below).

Rules:
  * Be concise and specific; cite page and location when helpful.
  * If something is not visible or cannot be determined, say so plainly. \
Never guess or fabricate.
  * For "is X present" questions (signature, seal, stamp, photo, watermark), \
answer Yes / No / Unclear first, then describe where it appears.
  * Do not give legal conclusions; you only describe what is visible and \
flag concerns for human review.
"""


# --------------------------------------------------------------------------- #
# File preprocessing
# --------------------------------------------------------------------------- #
def _normalize_image(img: Image.Image) -> bytes:
    """Fix orientation, flatten to RGB, cap resolution, return PNG bytes."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _pdf_to_images(data: bytes) -> list[Image.Image]:
    """Render PDF pages to PIL images using PyMuPDF."""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise SentryDocError("This PDF is corrupted. Re-export it and try again.")
    try:
        if doc.needs_pass:
            raise SentryDocError("This PDF is password-protected. Remove the password and retry.")
        if doc.page_count == 0:
            raise SentryDocError("The PDF has no pages.")
        pages: list[Image.Image] = []
        for i in range(min(doc.page_count, MAX_PDF_PAGES)):
            pix = doc[i].get_pixmap(dpi=PDF_DPI)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))))
        return pages
    except SentryDocError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("PDF rendering failed")
        raise SentryDocError(f"Could not read this PDF ({exc}).")
    finally:
        doc.close()


def preprocess_upload(name: str, data: bytes) -> list[bytes]:
    """Turn an uploaded PDF/image into a list of normalized PNG page images."""
    if not data:
        raise SentryDocError("The uploaded file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise SentryDocError(f"File is larger than {MAX_FILE_MB} MB. Upload a smaller file.")

    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if ext == "pdf" or data[:5] == b"%PDF-":
        return [_normalize_image(p) for p in _pdf_to_images(data)]

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return [_normalize_image(img)]
    except (UnidentifiedImageError, OSError):
        raise SentryDocError("This file is not a valid PNG or JPG image.")


# --------------------------------------------------------------------------- #
# Gemini service layer
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


@st.cache_data(ttl=600, show_spinner=False)
def list_models(api_key: str) -> list[str]:
    """Return text/vision Gemini models this key can call (empty list on failure)."""
    try:
        names: set[str] = set()
        for m in get_client(api_key).models.list():
            actions = getattr(m, "supported_actions", None) or []
            name = (m.name or "").removeprefix("models/")
            if (
                "generateContent" in actions
                and name.startswith("gemini")
                and not any(w in name for w in EXCLUDED_MODEL_WORDS)
            ):
                names.add(name)
        return sorted(names)
    except Exception:  # noqa: BLE001 - fall back to a manual text box
        log.exception("Could not list models")
        return []


def pick_default_model(models: list[str]) -> int:
    """Index of the best default: env var, then preferred list, then any flash-lite/flash."""
    wanted = [os.getenv("GEMINI_MODEL", "")] + PREFERRED_MODELS
    for name in wanted:
        if name in models:
            return models.index(name)
    for key in ("flash-lite", "flash"):
        for i, name in enumerate(models):
            if key in name and "preview" not in name:
                return i
    return 0


def _page_parts(pages: list[bytes]) -> list[types.Part]:
    parts: list[types.Part] = []
    for i, png in enumerate(pages, start=1):
        parts.append(types.Part.from_text(text=f"[Page {i}]"))
        parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))
    return parts


def _explain_auth_error(code: int | None, raw: str) -> str:
    """Turn an auth-type API error into plain steps the user can follow."""
    upper = raw.upper()
    if "ACCESS_TOKEN_TYPE_UNSUPPORTED" in upper:
        hint = (
            "Google accepted the key format but this SDK/key combination is not supported. "
            "Run `pip install -U google-genai`, create a NEW key in Google AI Studio, "
            "and paste it again."
        )
    elif "API KEY NOT VALID" in upper or "API_KEY_INVALID" in upper:
        hint = (
            "The key text is wrong or incomplete. Create a new key in AI Studio and use "
            "its copy button instead of selecting the text by hand."
        )
    elif "PERMISSION_DENIED" in upper or code == 403:
        hint = (
            "The key is valid but not allowed to use this model/API. Check that the "
            "Generative Language API is enabled for its project."
        )
    else:
        hint = "Check the key and the model name in Settings."
    return f"Gemini rejected the request (HTTP {code}). {hint}\n\nDetails: {raw}"


def _generate(client: genai.Client, **kwargs: Any) -> types.GenerateContentResponse:
    """generate_content with exponential backoff on transient API errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            raw = str(exc)
            if code in (400, 401, 403):
                raise SentryDocError(_explain_auth_error(code, raw))
            if code == 404:
                raise SentryDocError(
                    "Model not found or retired. Open Settings and pick another model from the dropdown."
                )
            if code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                wait = 2**attempt
                log.warning("Gemini error %s, retrying in %ss", code, wait)
                time.sleep(wait)
                continue
            if code == 429:
                raise SentryDocError("Rate limit reached. Wait a minute and try again.")
            raise SentryDocError(f"Gemini API error ({code}): {raw}")
        except Exception as exc:  # noqa: BLE001 - network failures etc.
            if attempt < MAX_RETRIES:
                time.sleep(2**attempt)
                continue
            raise SentryDocError(f"Could not reach Gemini: {exc}")
    raise SentryDocError("Gemini request failed.")  # pragma: no cover


def test_connection(client: genai.Client, model: str) -> str:
    """Tiny text-only call used by the 'Test connection' button."""
    resp = _generate(client, model=model, contents="Reply with the single word: OK")
    return (resp.text or "").strip() or "(empty reply)"


def analyze_document(client: genai.Client, model: str, pages: list[bytes]) -> dict:
    """Run structured multimodal analysis; returns the spec's JSON shape."""
    contents = [
        *_page_parts(pages),
        types.Part.from_text(text=ANALYSIS_USER_PROMPT.format(n=len(pages))),
    ]
    config = types.GenerateContentConfig(
        system_instruction=ANALYSIS_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=DocumentAnalysis,
        temperature=0.1,
    )
    resp = _generate(client, model=model, contents=contents, config=config)

    parsed = getattr(resp, "parsed", None)
    try:
        result = (
            parsed
            if isinstance(parsed, DocumentAnalysis)
            else DocumentAnalysis.model_validate_json(resp.text or "")
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Unparseable model output: %s", (resp.text or "")[:500])
        raise SentryDocError(
            "The model returned a response that could not be parsed (it may have been "
            "blocked or truncated). Try again or use a clearer scan."
        ) from exc

    score = max(0, min(100, int(result.authenticity_score)))
    return {
        "document_type": result.document_type,
        "extracted_fields": {f.name: f.value for f in result.extracted_fields},
        "authenticity_score": score,
        "tampering_detected": result.tampering_detected,
        "flagged_reasons": result.flagged_reasons,
        "summary": result.summary,
    }


def answer_question(
    client: genai.Client,
    model: str,
    pages: list[bytes],
    analysis: dict | None,
    history: list[dict],
    question: str,
) -> str:
    """Stateless multi-turn chat: pages + report are replayed with each call."""
    context = "Document pages are attached."
    if analysis:
        context += "\nPrior verification report:\n" + json.dumps(analysis, indent=2)

    contents = [
        types.Content(role="user", parts=[*_page_parts(pages), types.Part.from_text(text=context)]),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text="Understood. Ask me anything about this document.")],
        ),
    ]
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    config = types.GenerateContentConfig(system_instruction=CHAT_SYSTEM_PROMPT, temperature=0.2)
    resp = _generate(client, model=model, contents=contents, config=config)
    text = (resp.text or "").strip()
    if not text:
        raise SentryDocError(
            "The model returned no answer (the response may have been blocked). Rephrase and retry."
        )
    return text

RISK_META = {
    "error": {"label": "HIGH RISK", "color": "#ef4444", "icon": "&#9888;", "bg": "rgba(239,68,68,.12)"},
    "warning": {"label": "NEEDS REVIEW", "color": "#f59e0b", "icon": "&#9201;", "bg": "rgba(245,158,11,.12)"},
    "success": {"label": "LOW RISK", "color": "#10b981", "icon": "&#10003;", "bg": "rgba(16,185,129,.12)"},
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
#MainMenu, footer { visibility: hidden; }

/* ---------- hero ---------- */
.hero {
    position: relative; overflow: hidden; border-radius: 22px; padding: 34px 38px; margin-bottom: 18px;
    background: radial-gradient(900px 300px at 85% -20%, rgba(124,58,237,.55), transparent 60%),
                radial-gradient(700px 300px at 0% 120%, rgba(14,165,233,.45), transparent 60%),
                linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid rgba(255,255,255,.08); box-shadow: 0 18px 50px -18px rgba(30,27,75,.8);
}
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 800; color: #fff; letter-spacing: -.02em; }
.hero h1 span { background: linear-gradient(90deg,#38bdf8,#a78bfa,#f472b6); -webkit-background-clip: text; color: transparent; }
.hero p { margin: 8px 0 18px; color: #cbd5e1; font-size: 1.02rem; max-width: 640px; }
.chip { display: inline-block; padding: 5px 13px; margin: 0 8px 6px 0; border-radius: 999px; font-size: .78rem; font-weight: 600;
        color: #e2e8f0; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); backdrop-filter: blur(6px); }

/* ---------- stepper ---------- */
.steps { display: flex; gap: 10px; margin: 6px 0 20px; }
.step { flex: 1; display: flex; align-items: center; gap: 10px; padding: 12px 16px; border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.06); font-weight: 600; font-size: .9rem; opacity: .6; }
.step .dot { width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; font-size: .8rem;
             background: rgba(128,128,128,.25); }
.step.active { opacity: 1; border-color: #7c3aed; background: rgba(124,58,237,.10); }
.step.active .dot { background: #7c3aed; color: #fff; }
.step.done { opacity: 1; border-color: rgba(16,185,129,.5); }
.step.done .dot { background: #10b981; color: #fff; }

/* ---------- cards / verdict ---------- */
.verdict { display: flex; align-items: center; gap: 16px; padding: 18px 22px; border-radius: 16px; margin-bottom: 16px; border: 1px solid; }
.verdict .ico { font-size: 1.6rem; width: 46px; height: 46px; border-radius: 12px; display: grid; place-items: center; color: #fff; }
.verdict .tag { font-size: .72rem; font-weight: 800; letter-spacing: .12em; }
.verdict .msg { font-size: 1.02rem; font-weight: 600; margin-top: 2px; }

.stat { padding: 16px 18px; border-radius: 16px; border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.06); height: 100%; }
.stat .k { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; opacity: .65; font-weight: 700; }
.stat .v { font-size: 1.35rem; font-weight: 800; margin-top: 6px; line-height: 1.2; word-break: break-word; }

.ring-wrap { display: flex; justify-content: center; padding: 6px 0; }
.ring { position: relative; width: 150px; height: 150px; }
.ring svg { width: 100%; height: 100%; transform: rotate(-90deg); }
.ring circle { fill: none; stroke-width: 10; }
.ring-bg { stroke: rgba(128,128,128,.22); }
.ring-fg { stroke-linecap: round; animation: ringfill 1.2s ease-out; }
@keyframes ringfill { from { stroke-dashoffset: 339.3; } }
.ring-label { position: absolute; inset: 0; display: grid; place-content: center; text-align: center; }
.ring-label b { font-size: 2.2rem; font-weight: 800; line-height: 1; }
.ring-label span { font-size: .75rem; opacity: .6; font-weight: 600; }

.summary { padding: 16px 20px; border-radius: 16px; border-left: 4px solid #7c3aed; background: rgba(124,58,237,.07); margin: 14px 0 6px; line-height: 1.6; }

.flag { display: flex; gap: 12px; align-items: flex-start; padding: 12px 14px; border-radius: 12px; margin-bottom: 10px;
        background: rgba(245,158,11,.10); border: 1px solid rgba(245,158,11,.35); }
.flag .n { min-width: 26px; height: 26px; border-radius: 8px; background: #f59e0b; color: #fff; display: grid; place-items: center; font-weight: 800; font-size: .8rem; }
.clean { padding: 22px; text-align: center; border-radius: 14px; background: rgba(16,185,129,.10); border: 1px dashed rgba(16,185,129,.5); font-weight: 600; }

.disclaimer { font-size: .8rem; opacity: .65; margin-top: 14px; text-align: center; }
.empty { text-align: center; padding: 50px 20px; border-radius: 20px; border: 2px dashed rgba(128,128,128,.35); background: rgba(128,128,128,.04); }
.empty .big { font-size: 2.6rem; }
.empty h3 { margin: 6px 0 4px; }

/* ---------- buttons / sidebar ---------- */
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#7c3aed,#2563eb); border: 0; color: #fff; font-weight: 700;
    padding: .7rem 1rem; border-radius: 12px; box-shadow: 0 8px 22px -8px rgba(124,58,237,.7); transition: transform .15s;
}
div.stButton > button[kind="primary"]:hover { transform: translateY(-2px); }
div.stButton > button { border-radius: 10px; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.2); }
.side-title { display: flex; align-items: center; gap: 10px; font-size: 1.15rem; font-weight: 800; margin-bottom: 4px; }
.side-title .badge { background: linear-gradient(135deg,#7c3aed,#2563eb); color: #fff; width: 32px; height: 32px; border-radius: 10px; display: grid; place-items: center; }
[data-testid="stFileUploader"] section { border-radius: 16px; border: 2px dashed rgba(124,58,237,.5); background: rgba(124,58,237,.05); }
[data-testid="stTabs"] button { font-weight: 600; }
</style>
"""


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "doc_id": None,
        "doc_name": None,
        "pages": [],
        "analysis": None,
        "chat": [],
        "pending_q": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_document(doc_id: str, name: str, pages: list[bytes]) -> None:
    st.session_state.update(
        doc_id=doc_id, doc_name=name, pages=pages, analysis=None, chat=[], pending_q=None
    )


def verdict(analysis: dict) -> tuple[str, str]:
    score, tampered = analysis["authenticity_score"], analysis["tampering_detected"]
    if tampered or score < 50:
        return "error", "High risk - likely altered or forged. Escalate to manual review."
    if score < 80:
        return "warning", "Needs review - some anomalies were found."
    return "success", "Low risk - no significant anomalies found."


# --------------------------------------------------------------------------- #
# Small HTML helpers
# --------------------------------------------------------------------------- #
def hero() -> None:
    st.markdown(
        """
<div class="hero">
  <h1>&#128737; Sentry<span>Doc</span> AI</h1>
  <p>Upload a certificate, marksheet or ID and get an AI-powered tamper check in seconds -
     fonts, seals, alignment and internal consistency, all inspected for you.</p>
  <span class="chip">&#128270; Forgery detection</span>
  <span class="chip">&#129534; Field extraction</span>
  <span class="chip">&#128172; Ask the document</span>
  <span class="chip">&#128196; PDF / PNG / JPG</span>
</div>
""",
        unsafe_allow_html=True,
    )


def stepper(has_doc: bool, has_report: bool) -> None:
    def cls(done: bool, active: bool) -> str:
        return "done" if done else ("active" if active else "")

    s1 = cls(has_doc, not has_doc)
    s2 = cls(has_report, has_doc and not has_report)
    s3 = cls(False, has_report)
    st.markdown(
        f"""
<div class="steps">
  <div class="step {s1}"><div class="dot">{'&#10003;' if has_doc else '1'}</div>Upload document</div>
  <div class="step {s2}"><div class="dot">{'&#10003;' if has_report else '2'}</div>Verify with AI</div>
  <div class="step {s3}"><div class="dot">3</div>Review &amp; ask questions</div>
</div>
""",
        unsafe_allow_html=True,
    )


def score_ring(score: float, color: str) -> str:
    score = max(0.0, min(100.0, float(score)))
    circumference = 339.29
    offset = circumference * (1 - score / 100)
    return (
        '<div class="ring-wrap"><div class="ring">'
        '<svg viewBox="0 0 120 120">'
        '<circle class="ring-bg" cx="60" cy="60" r="54"/>'
        f'<circle class="ring-fg" cx="60" cy="60" r="54" stroke="{color}" '
        f'stroke-dasharray="{circumference}" stroke-dashoffset="{offset:.1f}"/>'
        "</svg>"
        f'<div class="ring-label"><b>{score:.0f}</b><span>/ 100</span></div>'
        "</div></div>"
    )


def stat_card(label: str, value: str, color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    return f'<div class="stat"><div class="k">{escape(label)}</div><div class="v"{style}>{escape(str(value))}</div></div>'


def empty_state() -> None:
    st.markdown(
        """
<div class="empty">
  <div class="big">&#128196;</div>
  <h3>No document yet</h3>
  <div style="opacity:.7">Drop a PDF, PNG or JPG above to preview it here and run a verification.</div>
</div>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def render_report(analysis: dict) -> None:
    level, message = verdict(analysis)
    meta = RISK_META[level]
    score = analysis["authenticity_score"]
    tampered = analysis["tampering_detected"]

    # Verdict banner
    st.markdown(
        f"""
<div class="verdict" style="background:{meta['bg']};border-color:{meta['color']}66">
  <div class="ico" style="background:{meta['color']}">{meta['icon']}</div>
  <div>
    <div class="tag" style="color:{meta['color']}">{meta['label']}</div>
    <div class="msg">{escape(message)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Score ring + stat cards
    ring_col, stats_col = st.columns([1, 1.5], gap="medium")
    with ring_col:
        st.markdown(score_ring(score, meta["color"]), unsafe_allow_html=True)
    with stats_col:
        a, b = st.columns(2)
        a.markdown(
            stat_card("Tampering", "Detected" if tampered else "Not detected", "#ef4444" if tampered else "#10b981"),
            unsafe_allow_html=True,
        )
        b.markdown(stat_card("Document type", analysis["document_type"]), unsafe_allow_html=True)
        st.write("")
        c, d = st.columns(2)
        c.markdown(stat_card("Flags raised", str(len(analysis["flagged_reasons"]))), unsafe_allow_html=True)
        d.markdown(stat_card("Fields found", str(len(analysis["extracted_fields"]))), unsafe_allow_html=True)

    if analysis.get("summary"):
        st.markdown(f'<div class="summary">{escape(analysis["summary"])}</div>', unsafe_allow_html=True)

    # Details in tabs
    flags = analysis["flagged_reasons"]
    fields = analysis["extracted_fields"]
    t_fields, t_flags, t_json = st.tabs(
        [f":material/badge: Extracted fields ({len(fields)})", f":material/flag: Flagged reasons ({len(flags)})", ":material/data_object: Raw JSON"]
    )
    with t_fields:
        if fields:
            st.dataframe(
                [{"Field": k, "Value": v} for k, v in fields.items()],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No fields could be extracted.")
    with t_flags:
        if flags:
            for i, reason in enumerate(flags, 1):
                st.markdown(
                    f'<div class="flag"><div class="n">{i}</div><div>{escape(str(reason))}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="clean">&#10003; Nothing flagged.</div>', unsafe_allow_html=True)
    with t_json:
        st.json(analysis)

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(analysis, indent=2),
        file_name="sentrydoc_report.json",
        mime="application/json",
        icon=":material/download:",
        width="stretch",
    )
    st.markdown(
        '<div class="disclaimer">The score is an AI risk signal for triage, not proof of forgery or authenticity.</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Chat (sidebar)
# --------------------------------------------------------------------------- #
SUGGESTIONS = ["Is the signature present?", "Which seal is on this page?", "Any mismatched fonts?"]


def render_chat(client: genai.Client | None, model: str) -> None:
    sb = st.sidebar
    sb.markdown(
        '<div class="side-title"><div class="badge">&#128172;</div>Ask about this document</div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.pages:
        sb.info("Upload a document to start asking questions.", icon=":material/upload_file:")
        return
    if client is None:
        sb.warning("Add your Gemini API key to enable chat.", icon=":material/key:")
        return

    sb.caption(f"Document: **{st.session_state.doc_name}**")

    box = sb.container(height=400, border=True)
    with box:
        if not st.session_state.chat:
            st.caption("Quick questions:")
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/verified_user:"):
                st.markdown(msg["content"])

    # Quick-question buttons (only before the first message)
    if not st.session_state.chat:
        for i, text in enumerate(SUGGESTIONS):
            if sb.button(text, key=f"sug_{i}", width="stretch"):
                st.session_state.pending_q = text
                st.rerun()

    question = st.session_state.pending_q or sb.chat_input("Ask a question about the document")
    if question:
        st.session_state.pending_q = None
        history = list(st.session_state.chat)
        st.session_state.chat.append({"role": "user", "content": question})
        try:
            with st.spinner("Reading the document..."):
                answer = answer_question(
                    client, model, st.session_state.pages, st.session_state.analysis, history, question
                )
        except SentryDocError as exc:
            answer = f"Sorry, I couldn't answer that. {exc}"
        st.session_state.chat.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat and sb.button("Clear chat", icon=":material/delete:", width="stretch"):
        st.session_state.chat = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def render_settings() -> tuple[genai.Client | None, str]:
    """Settings box: a typed key overrides the .env/environment key."""
    env_key = clean_key(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    with st.expander("Settings", expanded=not env_key, icon=":material/tune:"):
        typed = clean_key(
            st.text_input(
                "Gemini API key",
                type="password",
                placeholder="Leave empty to use GEMINI_API_KEY from .env",
            )
        )
        api_key = typed or env_key

        models = list_models(api_key) if api_key else []
        if models:
            model = st.selectbox(
                "Gemini model",
                models,
                index=pick_default_model(models),
                help="Models your key can use right now. Flash-Lite models usually have the most free requests.",
            )
        else:
            model = st.text_input("Gemini model", value=DEFAULT_MODEL).strip() or DEFAULT_MODEL
            if api_key:
                st.caption("Could not load the model list, so type a model name manually.")

        if api_key:
            source = "Settings box" if typed else ".env / environment"
            st.caption(
                f":material/lock: Using key from {source} - {len(api_key)} characters, "
                f"starts with '{api_key[:3]}', ends with '{api_key[-3:]}'."
            )
        client = get_client(api_key) if api_key else None

        if st.button("Test connection", disabled=client is None, icon=":material/wifi_tethering:"):
            try:
                with st.spinner("Contacting Gemini..."):
                    reply = test_connection(client, model)
                st.success(f"Connected. Model replied: {reply}")
            except SentryDocError as exc:
                st.error(str(exc))
    return client, model


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="SentryDoc AI",
        page_icon=":material/verified_user:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    hero()
    client, model = render_settings()

    uploaded = st.file_uploader("Document (PDF, PNG or JPG)", type=ALLOWED_TYPES)

    if uploaded is not None:
        data = uploaded.getvalue()
        doc_id = hashlib.sha256(data).hexdigest()
        if doc_id != st.session_state.doc_id:
            try:
                with st.spinner("Preparing document..."):
                    pages = preprocess_upload(uploaded.name, data)
                reset_document(doc_id, uploaded.name, pages)
            except SentryDocError as exc:
                st.session_state.update(doc_id=None, pages=[], analysis=None, chat=[])
                st.error(str(exc))
    else:
        st.session_state.update(doc_id=None, pages=[], analysis=None, chat=[], pending_q=None)

    pages = st.session_state.pages
    stepper(bool(pages), bool(st.session_state.analysis))

    if pages:
        preview, report = st.columns([1, 1.4], gap="large")
        with preview:
            with st.container(border=True):
                st.markdown(f"**:material/description: {escape(st.session_state.doc_name or 'Document')}**")
                if len(pages) > 1:
                    tabs = st.tabs([f"Page {n}" for n in range(1, len(pages) + 1)])
                    for tab, png in zip(tabs, pages):
                        tab.image(png, width="stretch")
                else:
                    st.image(pages[0], width="stretch")
                if uploaded is not None and uploaded.name.lower().endswith(".pdf"):
                    st.caption(f"Analyzing up to the first {MAX_PDF_PAGES} pages.")

        with report:
            if st.button(
                "Verify document",
                type="primary",
                disabled=client is None,
                icon=":material/verified_user:",
                width="stretch",
            ):
                try:
                    with st.spinner("Inspecting fonts, seals, alignment and consistency..."):
                        st.session_state.analysis = analyze_document(client, model, pages)
                    st.rerun()
                except SentryDocError as exc:
                    st.error(str(exc))
            if client is None:
                st.info("Add your Gemini API key in Settings to verify documents.", icon=":material/key:")
            if st.session_state.analysis:
                render_report(st.session_state.analysis)
            elif client is not None:
                st.caption("Press **Verify document** to run the tamper analysis.")
    else:
        empty_state()

    render_chat(client, model)


if __name__ == "__main__":
    main()