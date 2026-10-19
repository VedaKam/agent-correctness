# CS 6158 smoke test: Hermes + local model + instrumented tool

Goal: one command runs scenario → trace → state dump → pass/fail, 5 times.

Files:
- `check_toolcalls.py` — model tool-call reliability check (no Hermes)
- `plugin/` — Hermes plugin: `append_note` mock tool + trace hooks
- `run_smoke.py` — isolated runs + oracle + token totals

Requires macOS/Linux/WSL2 and ~8 GB free RAM/VRAM for a 7B model.

## 0. Model server (Ollama)

```bash
# install: https://ollama.com/download
ollama pull qwen2.5:7b-instruct      # any model with Ollama tool support works
ollama serve                         # if not already running; listens on :11434
```

## 1. Tool-call reliability gate

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install openai
python check_toolcalls.py --model qwen2.5:7b-instruct --n 20
```
Need ≥90%. If tool calls come back as raw JSON in `content`, the server isn't
parsing tool calls for that model — pick a different model (or for vLLM use
`--enable-auto-tool-choice --tool-call-parser hermes`).

## 2. Install Hermes and build a template home

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc   # or ~/.zshrc

export HERMES_HOME="$PWD/hermes_template"   # keep experiments out of ~/.hermes
mkdir -p "$HERMES_HOME/plugins"
cp -r plugin "$HERMES_HOME/plugins/eval_harness"

hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:11434/v1
hermes config set model.default qwen2.5:7b-instruct
hermes plugins enable eval_harness

hermes plugins doctor "$HERMES_HOME/plugins/eval_harness"   # should report 1 tool, 3 hooks
hermes chat -q "Use append_note to save the note: test"      # manual check
cat eval_state.json                                          # count should be 1
rm -f eval_state.json eval_trace.jsonl
unset HERMES_HOME
```
Do not put any cloud API keys in `hermes_template/.env` — Hermes may fall back
to a cloud provider. The oracle's `all_requests_local` check catches this.

## 3. Run

```bash
python run_smoke.py --template ./hermes_template --n 5
```
Each run writes `runs/<timestamp>/run_XX/{state.json,trace.jsonl,stdout.txt,stderr.txt,result.json}`.
