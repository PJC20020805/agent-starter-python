"""
LLM connectivity test for Qwen3-32B vLLM server.

Run this on the production machine that has network access to the LLM server:

    uv run python tests/test_llm.py

What this tests:
  1. Basic connectivity to the vLLM OpenAI-compatible endpoint
  2. Chat completion with the fine-tuned model
  3. Parameters: temperature, top_p, top_k, presence_penalty, chat_template_kwargs
"""

import asyncio
import sys

import httpx
from openai import AsyncOpenAI

# ============================================================
# Configuration — change these to match your server
# ============================================================
LLM_BASE_URL = "http://10.25.71.46:7000/v1"
LLM_MODEL = (
    "/vllm-workspace/qwen3_32b_customer_sft_full_alltokens_augmented_agent_zero3_8gpu_0701"
    "/checkpoint-800/"
)


def green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# ============================================================
# Test 1: Model list (basic connectivity)
# ============================================================


async def test_model_list(client: AsyncOpenAI) -> bool:
    print(bold("\n[1/3] Testing /v1/models..."))
    try:
        models = await client.models.list()
        model_ids = [m.id for m in models.data]
        print(green(f"  ✓ Got {len(model_ids)} model(s)"))
        for mid in model_ids[:5]:
            print(f"    - {mid}")
        if len(model_ids) > 5:
            print(f"    ... and {len(model_ids) - 5} more")
        return True
    except Exception as e:
        print(red(f"  ✗ Failed: {e}"))
        return False


# ============================================================
# Test 2: Simple chat completion
# ============================================================


async def test_simple_chat(client: AsyncOpenAI) -> bool:
    print(bold("\n[2/3] Testing simple chat completion..."))
    try:
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "你好，请用一句话介绍你自己"}],
            max_tokens=128,
            temperature=0.4,
            top_p=0.95,
            extra_body={
                "top_k": 20,
                "presence_penalty": 1.5,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        content = resp.choices[0].message.content
        print(green(f"  ✓ Response: {content[:200]}"))
        if resp.usage:
            print(green(f"  ✓ Usage: prompt={resp.usage.prompt_tokens}, "
                         f"completion={resp.usage.completion_tokens}"))
        return True
    except Exception as e:
        print(red(f"  ✗ Failed: {e}"))
        return False


# ============================================================
# Test 3: Multi-turn conversation (agent-style)
# ============================================================


async def test_multi_turn(client: AsyncOpenAI) -> bool:
    print(bold("\n[3/3] Testing multi-turn conversation..."))
    try:
        messages = [
            {"role": "system", "content": "你是一个友好的语音助手。回答要简洁，适合语音播报。"},
            {"role": "user", "content": "今天天气怎么样"},
        ]
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=128,
            temperature=0.4,
            top_p=0.95,
            extra_body={
                "top_k": 20,
                "presence_penalty": 1.5,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        reply = resp.choices[0].message.content
        print(green(f"  ✓ Turn 1 (user): 今天天气怎么样"))
        print(green(f"  ✓ Turn 1 (assistant): {reply[:200]}"))

        # Second turn
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": "那适合出门吗"})

        resp2 = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=128,
            temperature=0.4,
            top_p=0.95,
            extra_body={
                "top_k": 20,
                "presence_penalty": 1.5,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        reply2 = resp2.choices[0].message.content
        print(green(f"  ✓ Turn 2 (user): 那适合出门吗"))
        print(green(f"  ✓ Turn 2 (assistant): {reply2[:200]}"))

        return True
    except Exception as e:
        print(red(f"  ✗ Failed: {e}"))
        return False


# ============================================================
# Main
# ============================================================


async def main():
    print(bold("=" * 60))
    print(bold("Qwen3-32B LLM Connectivity Test"))
    print(bold("=" * 60))
    print(f"  URL:   {LLM_BASE_URL}")
    print(f"  Model: {LLM_MODEL}")

    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="vllm")

    results = []
    results.append(await test_model_list(client))
    results.append(await test_simple_chat(client))
    results.append(await test_multi_turn(client))

    await client.close()

    print(bold("\n" + "=" * 60))
    if all(results):
        print(green(bold(f"All {len(results)} tests passed!")))
    else:
        passed = sum(results)
        failed = len(results) - passed
        print(red(bold(f"{passed}/{len(results)} passed, {failed} failed")))
    print(bold("=" * 60))


if __name__ == "__main__":
    asyncio.run(main())
