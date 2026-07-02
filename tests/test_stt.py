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
import os
import struct
import sys
import time
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


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Convert raw PCM bytes to WAV format."""
    data_size = len(pcm_data)
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(bits_per_sample // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
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
# Test 4: Real audio file prediction
# ============================================================


async def test_real_file(file_path: str) -> bool:
    print(bold(f"\n[4/4] Real file prediction: {file_path}"))

    # Read audio file
    if not os.path.exists(file_path):
        print(red(f"  ✗ File not found: {file_path}"))
        return False

    try:
        with open(file_path, "rb") as f:
            raw = f.read()
    except Exception as e:
        print(red(f"  ✗ Failed to read file: {e}"))
        return False

    # Convert to WAV if needed
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".mp3", ".m4a", ".ogg", ".flac", ".opus", ".wma"):
        print(f"  Converting {ext} → WAV via ffmpeg...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-i", file_path,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                "-f", "wav", "pipe:1",
                stdout=asyncio.subprocess.PIPE,
            )
            wav_bytes, stderr = await proc.communicate()
            if proc.returncode != 0:
                print(red(f"  ✗ ffmpeg failed (exit {proc.returncode})"))
                return False
            print(f"  ✓ Converted: {len(wav_bytes)} bytes WAV")
        except FileNotFoundError:
            print(red("  ✗ ffmpeg not found — install ffmpeg or use a WAV file"))
            return False
    elif ext == ".wav":
        wav_bytes = raw
        print(f"  ✓ Read WAV: {len(wav_bytes)} bytes")
    elif ext == ".pcm":
        wav_bytes = pcm_to_wav(raw, sample_rate=16000)
        print(f"  ✓ Converted PCM→WAV: {len(wav_bytes)} bytes")
    else:
        # Try as raw PCM
        print(f"  Unknown extension '{ext}', trying as raw PCM 16kHz mono s16le...")
        wav_bytes = pcm_to_wav(raw, sample_rate=16000)

    # Send to ASR
    b64_audio = base64.b64encode(wav_bytes).decode("utf-8")
    system_prompt = f"Language: Chinese\n{ASR_CONTEXT}"
    payload = {
        "model": ASR_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64_audio}"}},
                    {"type": "text", "text": "transcribe"},
                ],
            },
        ],
        "stream": False,
    }

    print(f"  Sending to {ASR_BASE_URL}...")
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0)
        ) as client:
            t0 = time.time()
            resp = await client.post(ASR_BASE_URL, json=payload)
            elapsed = time.time() - t0
            resp.raise_for_status()
            result = resp.json()

        choices = result.get("choices", [])
        if not choices:
            print(red("  ✗ No choices in response"))
            return False

        raw_text = choices[0].get("message", {}).get("content", "").strip()
        lang, clean_text = parse_asr_output(raw_text)

        print(green(f"  ✓ HTTP {resp.status_code} in {elapsed:.1f}s"))
        print(bold(f"\n  ┌─ ASR Result ─────────────────────────────"))
        print(bold(f"  │ Language: {lang or 'auto'}"))
        print(bold(f"  │ Raw:      {raw_text[:200]}"))
        print(bold(f"  │ Clean:    {clean_text[:200]}"))
        print(bold(f"  └──────────────────────────────────────────"))
        return True

    except httpx.ConnectError as e:
        print(red(f"  ✗ Connection failed: {e}"))
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
    file_path = sys.argv[1] if len(sys.argv) > 1 else None

    print(bold("=" * 60))
    print(bold("Qwen3-ASR STT Connectivity Test"))
    print(bold("=" * 60))

    # Tests 1 & 2: no network needed
    test_parse_asr_output()
    test_capabilities()

    if file_path:
        # Test 4: real file prediction
        ok = await test_real_file(file_path)
    else:
        # Test 3: connectivity check with silence
        ok = await test_real_asr_call()

    print(bold("\n" + "=" * 60))
    if ok:
        print(green(bold("All tests passed!")))
    else:
        print(red(bold("Some tests failed — check output above.")))
    print(bold("=" * 60))


if __name__ == "__main__":
    asyncio.run(main())
