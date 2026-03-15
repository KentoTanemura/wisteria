"""Minimal smoke test: send one chat completion request to vLLM server."""

import os
import sys
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

    print(f"Target: http://{host}:{port}/v1")
    print(f"Model:  {model_id}")
    print()

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "1文で自己紹介してください。"}],
            max_tokens=128,
        )
        content = resp.choices[0].message.content
        print(f"Response:\n  {content}")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
