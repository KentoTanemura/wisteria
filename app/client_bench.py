"""Simple latency benchmark: 5 sequential requests."""

import os
import sys
import time
from pathlib import Path


def main() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if key not in os.environ:
                    os.environ[key] = val

    from openai import OpenAI

    host = os.environ.get("HOST", "127.0.0.1")
    port = os.environ.get("PORT", "8000")
    api_key = os.environ.get("API_KEY", "EMPTY")
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-35B-A3B")

    client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)

    n_requests = 5
    prompts = [
        "日本の首都はどこですか？",
        "What is 2 + 3?",
        "Pythonの特徴を1文で述べてください。",
        "東京タワーの高さは？",
        "Explain GPU in one sentence.",
    ]

    print(f"Target: http://{host}:{port}/v1")
    print(f"Model:  {model_id}")
    print(f"Requests: {n_requests}")
    print()

    latencies: list[float] = []
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
            )
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            content = resp.choices[0].message.content or ""
            preview = content[:60].replace("\n", " ")
            print(f"  [{i+1}/{n_requests}] {elapsed:.3f}s  \"{preview}...\"")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{n_requests}] {elapsed:.3f}s  FAIL: {e}", file=sys.stderr)

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\nAverage latency: {avg:.3f}s ({len(latencies)}/{n_requests} succeeded)")
    else:
        print("\nAll requests failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
