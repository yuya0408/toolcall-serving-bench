"""Does llama.cpp's OpenAI-compatible endpoint honour `seed`? (docs/gpu-runbook.md, step 4c)

    ~/venv/bin/python scripts/gpu/check_llama_seed.py      # writes ~/out/checks/llama_seed.json

Starts llama-server once, sends the same sampled request (temperature 0.7,
seed 42) twice, and compares the two answers; a third request with another
seed shows whether sampling varies at all. If the two seeded answers differ,
runs on llama.cpp must not be read as reproducible draw for draw.
"""
from __future__ import annotations
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_cells  # noqa: E402

QUERY = "今年度の給与額の推移を、月次で出してください"


def main() -> int:
    import requests
    out = pathlib.Path.home() / "out" / "checks"
    out.mkdir(parents=True, exist_ok=True)
    cell = {"engine": "llama_cpp", "quantization": "gguf-bf16"}
    with (out / "llama_seed_server.log").open("w") as log:
        proc = subprocess.Popen(run_cells.server_command(cell), stdout=log, stderr=subprocess.STDOUT)
    try:
        if not run_cells._wait_ready("llama_cpp", proc):
            raise SystemExit("llama-server did not become ready; see ~/out/checks/llama_seed_server.log")

        def ask(seed: int) -> str:
            body = {"messages": [{"role": "user", "content": QUERY}], "temperature": 0.7,
                    "top_p": 0.8, "top_k": 20, "max_tokens": 128, "seed": seed, "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            r = requests.post(f"http://127.0.0.1:{run_cells.PORT['llama_cpp']}/v1/chat/completions",
                              json=body, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content") or ""

        a, b, other = ask(42), ask(42), ask(7)
        result = {"same_seed_identical": a == b, "other_seed_differs": other != a,
                  "answers": {"seed42_first": a, "seed42_second": b, "seed7": other}}
        (out / "llama_seed.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("same_seed_identical", "other_seed_differs")}))
        return 0
    finally:
        run_cells._stop(proc)


if __name__ == "__main__":
    sys.exit(main())
