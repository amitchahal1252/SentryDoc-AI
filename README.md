# SentryDoc AI

Real-time multimodal document verification and fraud screening, powered by the Gemini API.
Built for **hackCBS 9.0 (AI track)**.

Upload a certificate, marksheet, ID or contract (PDF, PNG or JPG). SentryDoc reads it, extracts the key fields, scores how trustworthy it looks, and draws boxes around anything suspicious.

## Features

- **Structured analysis:** document type, extracted fields, authenticity score (0-100), tampering flag and flagged reasons, returned as validated JSON.
- **Marked-up pages:** numbered boxes drawn on the pages where Gemini sees font mismatches, pasted regions, odd seals or signatures.
- **File metadata check:** warns if the file was made or edited with tools such as Photoshop or Canva, or if a PDF was modified after creation.
- **Interactive document chat:** ask questions like "Is the signature present?" with one-tap suggestions.
- **Reports:** download a Markdown report or the full JSON.
- **Model picker:** lists the Gemini models your API key can use, so retired model names never break the app.

## Tech stack

Python 3.10+, Streamlit, `google-genai` (Gemini API), PyMuPDF (PDF rendering), Pillow, Pydantic.

## Setup

```bash
git clone https://github.com/<your-username>/SentryDoc-AI.git
cd SentryDoc-AI
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

Then:

```bash
pip install -r requirements.txt
```

Get a free key at [Google AI Studio](https://aistudio.google.com/apikey) and save it in a file named `.env` next to `app.py`:

```
GEMINI_API_KEY=your-key-here
```

Run:

```bash
streamlit run app.py
```

You can also paste the key into the **Settings** box in the app instead of using `.env`.

## Try it

`samples/bad_contract_test.pdf` is a synthetic contract with deliberate flaws (impossible dates, wrong totals, mismatched names, pasted signature, odd seal, edited-looking metadata). All parties are fictional. Upload it and compare with a genuine document.

## How it works

1. The upload is validated and normalized (PDF pages rendered, images rotated and resized).
2. Gemini receives the page images with a forensic prompt and a Pydantic response schema, so the output is structured JSON.
3. File metadata is checked separately, with no AI.
4. Evidence boxes are validated and drawn on the pages.
5. The chat replays the pages and the report to Gemini on each question.

## Limitations

- The score is an AI-generated risk signal for triage, not proof of forgery or authenticity.
- It cannot check a document against the issuer's records.
- Box positions are model estimates and can be slightly off.
- Metadata signals are hints: honest tools leave the same traces.
- Free-tier Gemini quotas limit how many documents you can verify per day.

## Project structure

```
app.py                  Streamlit app (UI, Gemini calls, preprocessing)
requirements.txt        Python dependencies
.streamlit/config.toml  Theme
samples/                Synthetic test document
.env.example            Template for your API key
```

## Team

Add your team name and members here.
