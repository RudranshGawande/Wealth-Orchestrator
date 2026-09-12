"""Verify Gemini API connectivity using google-genai SDK + python-dotenv."""
from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from google import genai

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=REPO_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found.")
    print(f"Add it to {REPO_ROOT / '.env'} as: GEMINI_API_KEY=your-key-here")
    sys.exit(1)

MODELS_TO_TRY = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-1.5-flash"]
PROMPT = "Hello, Gemini!"

client = genai.Client(api_key=API_KEY)

last_error = None
for model in MODELS_TO_TRY:
    try:
        print(f"Trying model: {model}")
        response = client.models.generate_content(model=model, contents=PROMPT)
        print(f"SUCCESS with {model}")
        print("Response:")
        print(response.text)
        break
    except Exception as e:  # noqa: BLE001
        last_error = e
        print(f"FAILED with {model}: {e}")
else:
    print("All models failed.")
    if last_error:
        raise SystemExit(f"Gemini API test failed: {last_error}") from last_error
