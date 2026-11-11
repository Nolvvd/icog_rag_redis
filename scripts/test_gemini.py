"""Verification script for Google Gemini Flash API connectivity.
Tests the SDK and live endpoint, measures latency, and verifies key validity safely.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import requests

def test_gemini():
    print("=" * 70)
    print("GOOGLE GEMINI FLASH CONNECTIVITY & INFERENCE TEST")
    print("=" * 70)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        print("\n❌ STATUS: GEMINI_API_KEY is NOT set in .env")
        print("\nAction Required:")
        print("1. Go to Google AI Studio: https://aistudio.google.com/")
        print("2. Create an API key.")
        print(f"3. Open '{PROJECT_ROOT / '.env'}' and add:")
        print("   GEMINI_API_KEY=AIzaSyYourActualKeyHere")
        print("=" * 70)
        return False

    if api_key.startswith("your_"):
        print("\n❌ STATUS: GEMINI_API_KEY contains a placeholder value.")
        print("Please replace it with your actual key from https://aistudio.google.com/")
        print("=" * 70)
        return False

    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
    print(f"🔑 API Key Detected: {masked_key} (length: {len(api_key)} chars)")

    model = os.getenv("LLM_MODEL", "").strip() or "gemini-3-flash-preview"
    print(f"🤖 Target Model: {model}")
    print("📡 Sending test inference request to Google Gen AI...")

    # Method 1: Try Google Gen AI SDK
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=60
        )
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=model,
            contents="Respond with exactly: 'Gemini Flash connection confirmed for iCog Labs RAG application.'",
            config=config
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if response and response.text:
            print("\n✅ SUCCESS: Gemini Flash responded successfully via Google Gen AI SDK!")
            print(f"🤖 Verified Model: {model}")
            print(f"⏱️ Round-trip Latency: {latency_ms:.1f} ms")
            print(f"💬 Model Output: {response.text.strip()}")
            print("\nYour RAG application is now ready to use live Gemini Flash synthesis.")
            print("=" * 70)
            return True
    except Exception as sdk_err:
        print(f"SDK Attempt note: {sdk_err}. Trying REST endpoint...")

    # Method 2: REST Endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [{"text": "Hello! Please reply with exactly: 'Gemini Flash connection confirmed for iCog Labs RAG application.'"}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 60
        }
    }

    t0 = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=20)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if response.status_code == 200:
            data = response.json()
            reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            print("\n✅ SUCCESS: Gemini Flash responded successfully via REST API!")
            print(f"🤖 Verified Model: {model}")
            print(f"⏱️ Round-trip Latency: {latency_ms:.1f} ms")
            print(f"💬 Model Output: {reply}")
            print("\nYour RAG application is now ready to use live Gemini Flash synthesis.")
            print("=" * 70)
            return True
        else:
            print(f"\n❌ API Error: HTTP {response.status_code}")
            try:
                err_json = response.json()
                err_msg = err_json.get("error", {}).get("message", response.text)
                err_status = err_json.get("error", {}).get("status", "")
                print(f"   Status: {err_status}")
                print(f"   Message: {err_msg}")

                if response.status_code == 400:
                    print("\n💡 Troubleshooting: The API key format may be invalid or the model name is incorrect.")
                elif response.status_code == 403:
                    print("\n💡 Troubleshooting: Permission denied. Check that the Generative Language API is enabled for your project.")
                elif response.status_code == 404:
                    print(f"\n💡 Troubleshooting: Model '{model}' not found. Verify supported models using ModelService.ListModels.")
                elif response.status_code == 429:
                    print("\n💡 Troubleshooting: Rate limit or quota exceeded. Wait a few moments or check your Google AI Studio quota.")
            except Exception:
                print(f"   Raw response: {response.text}")
            print("=" * 70)
            return False

    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out after 20 seconds. Check your internet connection or proxy settings.")
        print("=" * 70)
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Network or connection error: {e}")
        print("=" * 70)
        return False

if __name__ == "__main__":
    test_gemini()
