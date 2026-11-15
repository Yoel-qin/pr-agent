"""One-shot measurement for #4876/#4877: cache tokens with and without the stagger.

Runs the real dispatch path — LiteLLMAIHandler.chat_completion for message assembly
and transport, run_chunk_batch for dispatch, add_token_usage for accounting — against
a live Anthropic-format endpoint (DeepSeek, which reports cache hits in usage). Each
phase gets a unique system prefix so it starts cache-cold. Lives on a measurement-only
branch; never merged.

Usage: DEEPSEEK_API_KEY=... uv run python measure_cache_stagger.py
"""

import asyncio
import os
import sys
import time
import uuid

sys.path.insert(0, ".")

import litellm

litellm.api_base = os.environ.get("DEEPSEEK_ANTHROPIC_BASE", "https://api.deepseek.com/anthropic")

from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.chunk_runner import run_chunk_batch
from pr_agent.algo.run_details import add_token_usage, get_run_details, init_run_details

MODEL = "anthropic/deepseek-chat"
N_CHUNKS = 4

_REVIEW_RULES = [
    "Focus on correctness first: logic errors, off-by-one mistakes, wrong operator precedence, and unreachable branches outrank style.",
    "Treat every unhandled exception path as a finding; note whether the caller can observe it or it is swallowed silently.",
    "Concurrency findings must name the shared state, the competing accesses, and the ordering guarantee that is missing.",
    "Security findings must describe the tainted data flow from source to sink before they mention the fix.",
    "Performance findings need a plausible hot path or a measured number; do not flag one-time setup costs on cold paths.",
    "API misuse findings should cite the contract that is violated, not just the call that looks unusual.",
    "Resource leaks are findings when the failure path can skip the release, not only the happy path.",
    "Prefer pointing at the smallest correct fix; a rewrite is a suggestion only when the design is the defect.",
    "Never propose a fix that changes public behavior without saying what breaks for callers.",
    "Mark each finding with severity: critical, high, medium, or low, and one line of impact for the user.",
    "Skip findings about generated code, vendored dependencies, and lockfiles unless they are security-critical.",
    "When the diff deletes a check, ask what invariant it protected and whether that invariant still holds elsewhere.",
    "When the diff adds a flag, check every caller for the default behavior change, not only the new ones.",
    "A test that only repeats the implementation is not coverage; name the behavior it pins.",
    "For each suggestion, keep the surrounding code's naming, comment density, and formatting conventions.",
    "Do not restate the diff back to the reviewer; describe only what the change does to behavior.",
    "If an error message is user-facing, check that it states the remedy, not just the failure.",
    "If a timeout or retry is added, check that the operation is idempotent before it can run twice.",
    "If configuration parsing changes, cover the string, list, and environment-supplied forms of the value.",
    "If a git provider changes, check the sibling providers for the same logic before finishing.",
]


def _system_prefix(tag: str) -> str:
    lines = [
        f"You are a senior reviewer for a large pull request. Session {tag}.",
        "The diff arrives split into chunks; each request below carries one chunk.",
        "Review each chunk independently under the following standing rules:",
    ]
    lines += [f"{i:02d}. {rule}" for i, rule in enumerate(_REVIEW_RULES, 1)]
    lines += [
        "Answer with a short verdict line and at most three findings.",
        "Findings outside the chunk's files are out of scope and must be dropped.",
        "Finish with one line summarizing what you would test first.",
    ]
    return "\n".join(lines)


_CODE_LINES = [
    "def resolve_target(config, name):\n    target = config.get('targets', {}).get(name)\n    if target is None:\n        raise KeyError(name)\n    return target",
    "class RetryPolicy:\n    def __init__(self, attempts=3, backoff=0.5):\n        self.attempts = attempts\n        self.backoff = backoff\n\n    def delay(self, attempt):\n        return self.backoff * (2 ** attempt)",
    "def merge_defaults(base, override):\n    merged = dict(base)\n    for key, value in override.items():\n        merged[key] = value\n    return merged",
    "async def drain(queue, sink):\n    while True:\n        item = await queue.get()\n        if item is _SENTINEL:\n            break\n        await sink(item)",
]


def _user_chunk(i: int) -> str:
    filler = "\n".join(f"    # chunk {i + 1} context line {n}" for n in range(30))
    body = _CODE_LINES[i % len(_CODE_LINES)]
    return f"Review chunk {i + 1} of {N_CHUNKS} of the pull request diff:\n\n```python\n{body}\n{filler}\n```\n"


def _usage_fields(usage) -> dict:
    def read(*names):
        for name in names:
            value = getattr(usage, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        if isinstance(usage, dict):
            for name in names:
                value = usage.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
        return 0

    return {
        "input": read("prompt_tokens", "input_tokens"),
        "output": read("completion_tokens", "output_tokens"),
        "cache_read": read(
            "cache_read_input_tokens", "_cache_read_input_tokens", "prompt_cache_hit_tokens"
        ),
        "cache_creation": read("cache_creation_input_tokens", "_cache_creation_input_tokens"),
    }


async def run_phase(name: str, stagger: bool) -> None:
    tag = uuid.uuid4().hex[:12]
    system = _system_prefix(tag)
    handler = LiteLLMAIHandler()
    details = init_run_details()
    per_call = []

    async def call(i: int):
        started = time.perf_counter()
        resp, _finish = await handler.chat_completion(MODEL, system, _user_chunk(i))
        fields = _usage_fields(getattr(resp, "usage", None))
        fields["seconds"] = round(time.perf_counter() - started, 2)
        per_call.append(fields)
        add_token_usage(getattr(resp, "usage", None))

    started = time.perf_counter()
    results = await run_chunk_batch(list(range(N_CHUNKS)), call, stagger_first=stagger)
    wall = time.perf_counter() - started

    errors = [r for r in results if isinstance(r, Exception)]
    print(f"\n=== {name} (stagger_first={stagger}, prefix tag {tag}) ===")
    print(f"shared system prefix: {len(system)} chars (~{len(system) // 4} tokens)")
    for fields in sorted(per_call, key=lambda f: f["seconds"], reverse=True):
        print(
            "  call: in={input} out={output} cache_read={cache_read} "
            "cache_creation={cache_creation} took={seconds}s".format(**fields)
        )
    totals = get_run_details() or details
    print(
        f"run_details totals: prompt_tokens={totals.prompt_tokens} "
        f"completion_tokens={totals.completion_tokens} "
        f"cache_read_tokens={totals.cache_read_tokens} "
        f"cache_creation_tokens={totals.cache_creation_tokens}"
    )
    print(f"wall: {wall:.2f}s  errors: {len(errors)}")
    for error in errors:
        print(f"  error: {type(error).__name__}: {error}")


async def main() -> None:
    await run_phase("baseline: plain parallel dispatch (main today)", stagger=False)
    await run_phase("staggered first chunk (this PR)", stagger=True)


if __name__ == "__main__":
    asyncio.run(main())
