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


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {"doc_id": None, "doc_name": None, "pages": [], "analysis": None, "chat": []}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_document(doc_id: str, name: str, pages: list[bytes]) -> None:
    st.session_state.update(doc_id=doc_id, doc_name=name, pages=pages, analysis=None, chat=[])


def verdict(analysis: dict) -> tuple[str, str]:
    score, tampered = analysis["authenticity_score"], analysis["tampering_detected"]
    if tampered or score < 50:
        return "error", "High risk - likely altered or forged. Escalate to manual review."
    if score < 80:
        return "warning", "Needs review - some anomalies were found."
    return "success", "Low risk - no significant anomalies found."


def render_report(analysis: dict) -> None:
    level, message = verdict(analysis)
    getattr(st, level)(message)

    c1, c2, c3 = st.columns(3)
    c1.metric("Authenticity score", f"{analysis['authenticity_score']} / 100")
    c2.metric("Tampering detected", "Yes" if analysis["tampering_detected"] else "No")
    c3.metric("Document type", analysis["document_type"])
    st.progress(analysis["authenticity_score"] / 100)

    if analysis.get("summary"):
        st.write(analysis["summary"])

    left, right = st.columns(2)
    with left:
        st.subheader("Extracted fields")
        fields = analysis["extracted_fields"]
        if fields:
            st.dataframe(
                [{"Field": k, "Value": v} for k, v in fields.items()],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No fields could be extracted.")
    with right:
        st.subheader("Flagged reasons")
        if analysis["flagged_reasons"]:
            for reason in analysis["flagged_reasons"]:
                st.warning(reason, icon=":material/flag:")
        else:
            st.success("Nothing flagged.", icon=":material/check_circle:")

    with st.expander("Raw JSON"):
        st.json(analysis)
    st.download_button(
        "Download report (JSON)",
        data=json.dumps(analysis, indent=2),
        file_name="sentrydoc_report.json",
        mime="application/json",
    )


def render_chat(client: genai.Client | None, model: str) -> None:
    st.sidebar.header("Ask about this document")
    if not st.session_state.pages:
        st.sidebar.info("Upload a document to start asking questions.")
        return
    if client is None:
        st.sidebar.warning("Add your Gemini API key to enable chat.")
        return

    box = st.sidebar.container(height=420)
    with box:
        if not st.session_state.chat:
            st.caption('Try: "Is the signature present?" or "Which seal is on this page?"')
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    question = st.sidebar.chat_input("Ask a question about the document")
    if question:
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

    if st.session_state.chat and st.sidebar.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()


def render_settings() -> tuple[genai.Client | None, str]:
    """Settings box: a typed key overrides the .env/environment key."""
    env_key = clean_key(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    with st.expander("Settings", expanded=not env_key):
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
                f"Using key from {source} - {len(api_key)} characters, "
                f"starts with '{api_key[:3]}', ends with '{api_key[-3:]}'."
            )
        client = get_client(api_key) if api_key else None

        if st.button("Test connection", disabled=client is None):
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
    st.set_page_config(page_title="SentryDoc AI", page_icon=":material/verified_user:", layout="wide")
    init_state()

    st.title("SentryDoc AI")
    st.caption("Upload a certificate, marksheet or ID to check it for signs of tampering.")

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
        st.session_state.update(doc_id=None, pages=[], analysis=None, chat=[])

    pages = st.session_state.pages
    if pages:
        preview, report = st.columns([1, 1.4], gap="large")
        with preview:
            if len(pages) > 1:
                tabs = st.tabs([f"Page {n}" for n in range(1, len(pages) + 1)])
                for tab, png in zip(tabs, pages):
                    tab.image(png, width="stretch")
            else:
                st.image(pages[0], width="stretch")
            if uploaded is not None and uploaded.name.lower().endswith(".pdf"):
                st.caption(f"Analyzing up to the first {MAX_PDF_PAGES} pages.")

        with report:
            if st.button("Verify document", type="primary", disabled=client is None):
                try:
                    with st.spinner("Inspecting fonts, seals, alignment and consistency..."):
                        st.session_state.analysis = analyze_document(client, model, pages)
                except SentryDocError as exc:
                    st.error(str(exc))
            if client is None:
                st.info("Add your Gemini API key in Settings to verify documents.")
            if st.session_state.analysis:
                render_report(st.session_state.analysis)
                st.caption(
                    "The score is an AI risk signal for triage, not proof of forgery or authenticity."
                )

    render_chat(client, model)


if __name__ == "__main__":
    main()