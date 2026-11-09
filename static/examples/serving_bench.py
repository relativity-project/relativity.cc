"""Measure a running SGLang-JAX or SGLang-TorchTPU server the way the models page does.

Start the server first, as the inference recipe or the PyTorch SGLang guide describes, then:

    python3 serving_bench.py single              # decode rate and first token, one request at a time
    python3 serving_bench.py single --sampled    # also the sampled settings, on the 5-token prompt
    python3 serving_bench.py batched             # 1, 4, 8 and 16 concurrent requests

Every request streams its output and generates exactly 128 tokens: greedy unless sampled, with
`ignore_eos` so that it doesn't stop early. Timing uses the streamed tokens' arrival times:

- time to first token: from sending the request to the first streamed token;
- decode rate: the tokens after the first, divided by the time from the first token to the last.

`single` sends each prompt twice to warm up, then five more times, and prints the medians. The
models page's decode column is the 5-token prompt's decode rate, and its first-token column is the
215-token prompt's time to first token.

`batched` sends n copies of the code prompt at once. Total throughput is all generated tokens
(128 per request) divided by the time from sending the requests to the last one finishing, so it
includes the prefills and waiting for a free slot; per user is the median of the requests' decode
rates, which start at each request's first token. Total is therefore less than n times per user.
Each batch size first warms up with two batches, and more (up to twelve) until two in a row have no
first token slower than two seconds, since the server compiles programs for new shapes on first
use; it then prints the medians of three batches.

Needs only Python 3's standard library.
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = {
    # 5, 19 and 215 tokens with Qwen3's tokenizer.
    "short": "The capital of France is",
    "code": "Write a Python function that returns the n-th Fibonacci number using "
    "memoization. Explain each step.\n\n",
    "long": " ".join(
        f"Section {i}: the committee reviewed the budget, the schedule and the staffing plan."
        for i in range(12)
    )
    + " Summarize the review in three sentences.",
}

SAMPLING = {
    "temperature 0.7": {"temperature": 0.7},
    "temperature 0.7, top_p 0.95": {"temperature": 0.7, "top_p": 0.95},
    "temperature 0.7, top_k 20": {"temperature": 0.7, "top_k": 20},
    "temperature 0.7, top_p 0.95, top_k 20": {"temperature": 0.7, "top_p": 0.95, "top_k": 20},
}


def generate(url, prompt, sampling=None):
    """One streamed 128-token request: (seconds to first token, decode tokens/s, tokens, text)."""
    params = {"temperature": 0, "max_new_tokens": 128, "ignore_eos": True, **(sampling or {})}
    body = json.dumps({"text": prompt, "stream": True, "sampling_params": params}).encode()
    request = urllib.request.Request(url + "/generate", body, {"Content-Type": "application/json"})
    start = time.perf_counter()
    stamps, text, last = [], "", 0
    with urllib.request.urlopen(request, timeout=3600) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            item = json.loads(line[6:])
            count = item["meta_info"]["completion_tokens"]
            if count > last:
                stamps.append((time.perf_counter(), count))
                last, text = count, item["text"]
    (t0, c0), (t1, c1) = stamps[0], stamps[-1]
    return t0 - start, (c1 - c0) / (t1 - t0), c1, text


def single(url, prompt, sampling=None, warmups=2, runs=5):
    for _ in range(warmups):
        generate(url, prompt, sampling)
    results = [generate(url, prompt, sampling) for _ in range(runs)]
    return {
        "first_token_ms": round(1000 * statistics.median(r[0] for r in results), 1),
        "decode_tok_s": round(statistics.median(r[1] for r in results), 1),
    }


def batch(url, n):
    start = time.perf_counter()
    with ThreadPoolExecutor(n) as pool:
        results = list(pool.map(lambda _: generate(url, PROMPTS["code"]), range(n)))
    wall = time.perf_counter() - start
    return (sum(r[2] for r in results) / wall, statistics.median(r[1] for r in results),
            max(r[0] for r in results))


def batched(url, n, runs=3, slow_first_token_s=2.0):
    warm = [batch(url, n) for _ in range(2)]
    while len(warm) < 12 and any(w[2] > slow_first_token_s for w in warm[-2:]):
        warm.append(batch(url, n))
    results = [batch(url, n) for _ in range(runs)]
    return {
        "total_tok_s": round(statistics.median(r[0] for r in results), 1),
        "per_user_tok_s": round(statistics.median(r[1] for r in results), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["single", "batched"])
    parser.add_argument("--url", default="http://127.0.0.1:31000")
    parser.add_argument("--sampled", action="store_true", help="single: also sample the 5-token prompt")
    parser.add_argument("--batches", default="1,4,8,16", help="batched: comma-separated request counts")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    if args.mode == "single":
        for name, prompt in PROMPTS.items():
            print(json.dumps({"prompt": name, **single(url, prompt)}), flush=True)
        if args.sampled:
            for name, sampling in SAMPLING.items():
                print(json.dumps({"prompt": "short", "sampling": name,
                                  **single(url, PROMPTS["short"], sampling)}), flush=True)
    else:
        for n in (int(b) for b in args.batches.split(",")):
            print(json.dumps({"requests": n, **batched(url, n)}), flush=True)


if __name__ == "__main__":
    main()
