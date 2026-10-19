"""Step 1: does this model emit correct, parseable tool calls on your server?

Talks to the model server directly (no Hermes) so you measure the model +
parser alone. Run before committing to a model.

    pip install openai
    python check_toolcalls.py --model qwen2.5:7b-instruct --n 20
"""
import argparse
import json

from openai import OpenAI

TOOL = {
    "type": "function",
    "function": {
        "name": "append_note",
        "description": "Append a single note to the user's notes file.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}
PROMPT = 'Save a note that says exactly: "buy milk"'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=20)
    a = p.parse_args()

    client = OpenAI(base_url=a.base_url, api_key="local")
    ok = 0
    for i in range(a.n):
        r = client.chat.completions.create(
            model=a.model,
            messages=[{"role": "user", "content": PROMPT}],
            tools=[TOOL],
            temperature=0.7,  # nonzero on purpose: we want the realistic rate
        )
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        good = (
            len(calls) == 1
            and calls[0].function.name == "append_note"
            and json.loads(calls[0].function.arguments).get("text", "").strip().lower() == "buy milk"
        )
        ok += good
        if not good:
            print(f"[{i}] FAIL  tool_calls={calls}  content={msg.content!r:.120}")
    rate = ok / a.n
    print(f"\n{ok}/{a.n} correct tool calls ({rate:.0%})")
    print("PASS: usable for experiments" if rate >= 0.9 else
          "FAIL: model/parser too unreliable — try another model or check the server's tool parser")


if __name__ == "__main__":
    main()
