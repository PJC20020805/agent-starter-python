"""
Qwen3-ASR custom STT plugin for LiveKit Agents.

Connects to a self-hosted Qwen3-ASR model served via vLLM
at /v1/chat/completions (OpenAI-compatible chat API).

The framework automatically wraps this batch STT with StreamAdapter + VAD
to segment audio before calling recognize().
"""

import base64
import logging
from typing import Optional

import httpx
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)

ASR_TEXT_TAG = "<asr_text>"

# Hotwords / context prompt for the ASR model
ASR_CONTEXT = (
    "分期乐,乐花卡,乐信,买吖,超市卡,轻松还,"
    "逾期,欠款,还款,本金,利息,滞纳金,罚息,手续费,服务费,账单,额度,待还,欠费,"
    "结清,减免,分期,延期,协商,代扣,扣款,转账,支付宝,微信,还款码,"
    "征信,失信,仲裁,律师函,起诉,法院,强制执行,法务,"
    "没钱,困难,发工资,月底,下个月,投诉,骚扰,不是本人,打错了,挂了"
)


def parse_asr_output(raw_text: str) -> tuple[Optional[str], str]:
    """Extract language and clean text from Qwen3-ASR response.

    The model wraps transcription in <asr_text> tags, optionally prefixed
    with a language hint like "Language: Chinese".
    """
    if not raw_text:
        return None, ""

    text = raw_text.strip()

    if ASR_TEXT_TAG in text:
        before_tag, after_tag = text.split(ASR_TEXT_TAG, 1)
        clean_text = after_tag.strip()
        language = None
        before_tag = before_tag.strip()
        if before_tag.lower().startswith("language:"):
            lang_str = before_tag[len("language:"):].strip(": ").strip()
            if lang_str:
                language = lang_str
        return language, clean_text

    # No tag — return raw text as-is
    return None, text


class QwenASR(stt.STT):
    """Custom STT plugin for Qwen3-ASR served via vLLM.

    Uses the OpenAI-compatible /v1/chat/completions endpoint with
    audio passed as a base64 data URL in the message content.

    Usage:
        stt = QwenASR(
            base_url="http://10.25.28.12:7000/v1/chat/completions",
            model="/vllm-workspace/Qwen3-ASR-1.7B",
            language="zh",
        )
    """

    def __init__(
        self,
        *,
        base_url: str = "http://10.25.28.12:7000/v1/chat/completions",
        model: str = "/vllm-workspace/Qwen3-ASR-1.7B",
        language: str = "zh",
        context: Optional[str] = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._base_url = base_url
        self._model = model
        self._language = language
        self._context = context if context is not None else ASR_CONTEXT
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "qwen3-asr-vllm"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=20, max_connections=100
                ),
            )
        return self._client

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        # 1. Convert LiveKit AudioBuffer → WAV bytes → base64
        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()
        b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

        # 2. Build the chat completions payload
        system_prompt = f"Language: Chinese\n{self._context}"
        payload: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{b64_audio}"
                            },
                        },
                        {"type": "text", "text": "transcribe"},
                    ],
                },
            ],
            "stream": False,
        }

        # 3. Call the vLLM server
        client = await self._get_client()
        resp = await client.post(self._base_url, json=payload)
        resp.raise_for_status()
        result = resp.json()

        # 4. Extract text from chat completions response
        choices = result.get("choices", [])
        raw_text = ""
        if choices:
            raw_text = choices[0].get("message", {}).get("content", "").strip()

        _, clean_text = parse_asr_output(raw_text)
        final_text = clean_text or raw_text

        logger.debug(
            "QwenASR recognized: raw=%r → clean=%r",
            raw_text[:80],
            final_text[:80],
        )

        # 5. Return SpeechEvent
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=final_text,
                    language=self._language,
                )
            ],
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
