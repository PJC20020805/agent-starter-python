"""
STT connectivity test for Qwen3-ASR vLLM server.

Run this on the production machine that has network access to the ASR server:

    uv run python tests/test_stt.py

What this tests:
  1. parse_asr_output() — text parsing (no network needed)
  2. QwenASR capabilities — verifies it correctly extends stt.STT
  3. Real ASR call — sends a short silent WAV to the vLLM server
"""

import asyncio
import base64
import io
import struct
import sys
import wave

import httpx

# Add src to path so we can import the plugin
sys.path.insert(0, "src")
from qwen_asr_stt import ASR_CONTEXT, QwenASR, parse_asr_output

# ============================================================
# Configuration — change these to match your server
# ============================================================
ASR_BASE_URL = "http://10.25.71.71:7003/v1/chat/completions"
ASR_MODEL = "/vllm-workspace/Qwen3-ASR-1.7B"

# ============================================================
# Helpers
# ============================================================


def make_silence_wav(duration_ms: int = 200, sample_rate: int = 16000) -> bytes:
    """Generate a short silent WAV file for testing."""
    num_samples = int(sample_rate * duration_ms / 1000)
    pcm = b"\x00\x00" * num_samples

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# ============================================================
# Test 1: parse_asr_output (no network)
# ============================================================


def test_parse_asr_output():
    print(bold("\n[1/3] Testing parse_asr_output..."))

    # Case 1: text with <asr_text> tag
    raw = "Language: Chinese\n<asr_text>你好世界"
    lang, text = parse_asr_output(raw)
    assert lang == "Chinese", f"Expected lang='Chinese', got {lang}"
    assert text == "你好世界", f"Expected '你好世界', got '{text}'"
    print(green(f"  ✓ Tagged output: lang={lang}, text='{text}'"))

    # Case 2: text without tag
    raw = "你好世界"
    lang, text = parse_asr_output(raw)
    assert text == "你好世界", f"Expected '你好世界', got '{text}'"
    print(green(f"  ✓ Plain output: text='{text}'"))

    # Case 3: empty
    lang, text = parse_asr_output("")
    assert text == "", f"Expected '', got '{text}'"
    print(green(f"  ✓ Empty output: text='{text}'"))

    print(green("  PASSED"))


# ============================================================
# Test 2: QwenASR capabilities (no network)
# ============================================================


def test_capabilities():
    print(bold("\n[2/3] Testing QwenASR capabilities..."))

    asr = QwenASR(base_url=ASR_BASE_URL, model=ASR_MODEL, language="zh")

    assert asr.model == ASR_MODEL, f"model mismatch: {asr.model}"
    assert asr.provider == "qwen3-asr-vllm", f"provider mismatch: {asr.provider}"
    assert asr.capabilities.streaming is False, "expected streaming=False"
    assert asr.capabilities.interim_results is False, "expected interim_results=False"

    print(green(f"  ✓ model: {asr.model}"))
    print(green(f"  ✓ provider: {asr.provider}"))
    print(green(f"  ✓ capabilities: streaming={asr.capabilities.streaming}, "
                 f"interim={asr.capabilities.interim_results}"))
    print(green("  PASSED"))


# ============================================================
# Test 3: Real ASR call (requires network)
# ============================================================


async def test_real_asr_call():
    print(bold("\n[3/3] Testing real ASR call..."))

    wav_bytes = make_silence_wav(duration_ms=200, sample_rate=16000)
    b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

    system_prompt = f"Language: Chinese\n{ASR_CONTEXT}"
    payload = {
        "model": ASR_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{b64_audio}"},
                    },
                    {"type": "text", "text": "transcribe"},
                ],
            },
        ],
        "stream": False,
    }

    print(f"  Connecting to {ASR_BASE_URL}...")
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0)
        ) as client:
            resp = await client.post(ASR_BASE_URL, json=payload)
            resp.raise_for_status()
            result = resp.json()

        choices = result.get("choices", [])
        if not choices:
            print(red("  ✗ No choices in response"))
            return False

        raw_text = choices[0].get("message", {}).get("content", "").strip()
        lang, clean_text = parse_asr_output(raw_text)

        print(green(f"  ✓ HTTP {resp.status_code}"))
        print(green(f"  ✓ Raw output: {raw_text[:100]}"))
        print(green(f"  ✓ Clean text: {clean_text[:100]}"))
        print(green("  PASSED"))
        return True

    except httpx.ConnectError as e:
        print(red(f"  ✗ Connection failed: {e}"))
        print(red("    Is the vLLM server running on port 7000?"))
        return False
    except httpx.HTTPStatusError as e:
        print(red(f"  ✗ HTTP error {e.response.status_code}: {e.response.text[:200]}"))
        return False
    except Exception as e:
        print(red(f"  ✗ Unexpected error: {e}"))
        return False


# ============================================================
# Main
# ============================================================


async def main():
    print(bold("=" * 60))
    print(bold("Qwen3-ASR STT Connectivity Test"))
    print(bold("=" * 60))

    # Tests 1 & 2: no network needed
    test_parse_asr_output()
    test_capabilities()

    # Test 3: needs network
    ok = await test_real_asr_call()

    print(bold("\n" + "=" * 60))
    if ok:
        print(green(bold("All tests passed!")))
    else:
        print(red(bold("Some tests failed — check output above.")))
    print(bold("=" * 60))


if __name__ == "__main__":
    asyncio.run(main())
