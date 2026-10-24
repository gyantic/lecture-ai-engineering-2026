"""第1回 宿題（発展）: 応答時間とトークン数を計測し、モデル・質問ごとに比較する。

使い方:
    uv run python scripts/bench_ask.py                       # 既定のモデル×質問×3回
    uv run python scripts/bench_ask.py --models gemini-3.5-flash --repeat 5
    uv run python scripts/bench_ask.py --out bench.csv       # 1回ごとの結果をCSVに保存
    uv run python scripts/bench_ask.py --burst 20 --models gemini-2.5-flash
                                                             # 間隔0で連打し、レート制限に当てる
    uv run python scripts/bench_ask.py --burst 30 --parallel 10   # 10本同時に送る

1回ごとに次を計測する（ストリーミングで受け取り、最初の文字が届くまでの時間も測る）。
  - TTFT: 送信してから最初のチャンクが届くまでの秒数（体感の「待ち時間」）
  - 応答時間: 送信してから最後のチャンクが届くまでの秒数
  - トークン数: 入力 / 出力 / 思考（thinking）/ 合計
429（レート制限）が返ったときは、レスポンスに含まれるクォータ名・上限値・再試行までの秒数を表示する。
"""

import argparse
import csv
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from google import genai
from google.genai import errors

# scripts/ から実行しても app パッケージを読めるようにする
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import get_api_key  # noqa: E402

DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
]

# 性質の違う3種類の質問（短答 / 一般的な説明 / 講座に固有）
QUESTIONS = {
    "short": "日本の首都はどこですか？一言で答えてください。",
    "lora": "LoRAとは何ですか？",
    "course": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
}


@dataclass
class Result:
    model: str
    question: str
    run: int
    status: int = 200
    ttft_s: float | None = None
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thoughts_tokens: int | None = None
    total_tokens: int | None = None
    error: str = ""
    quota: dict = field(default_factory=dict)


def quota_info(e: errors.APIError) -> dict:
    """429 のレスポンスから、どのクォータに当たったか・上限値・再試行までの時間を取り出す。"""
    info = {}
    details = (e.details or {}).get("error", {}).get("details", []) if isinstance(e.details, dict) else []
    for d in details:
        kind = d.get("@type", "")
        if kind.endswith("QuotaFailure"):
            v = (d.get("violations") or [{}])[0]
            info["quota_id"] = v.get("quotaId", "")
            info["quota_value"] = v.get("quotaValue", "")
        elif kind.endswith("RetryInfo"):
            info["retry_delay"] = d.get("retryDelay", "")
    return info


def measure(client: genai.Client, model: str, qid: str, run: int) -> Result:
    r = Result(model=model, question=qid, run=run)
    start = time.perf_counter()
    usage = None
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=QUESTIONS[qid]):
            if r.ttft_s is None:
                r.ttft_s = time.perf_counter() - start
            # 使用量は最後のチャンクに確定値が入る
            if chunk.usage_metadata:
                usage = chunk.usage_metadata
    except errors.APIError as e:
        r.status = e.code
        r.error = (e.message or "")[:120]
        if e.code == 429:
            r.quota = quota_info(e)
        return r
    r.latency_s = time.perf_counter() - start
    if usage:
        r.input_tokens = usage.prompt_token_count
        r.output_tokens = usage.candidates_token_count
        r.thoughts_tokens = usage.thoughts_token_count or 0
        r.total_tokens = usage.total_token_count
    return r


def print_line(r: Result) -> None:
    if r.status != 200:
        print(f"  {r.model:24} {r.question:7} #{r.run}  ERROR {r.status} {r.quota or r.error}")
        return
    print(
        f"  {r.model:24} {r.question:7} #{r.run}  "
        f"TTFT {r.ttft_s:5.2f}s  応答 {r.latency_s:5.2f}s  "
        f"入力 {r.input_tokens:4} / 出力 {r.output_tokens:5} / 思考 {r.thoughts_tokens:5} / 合計 {r.total_tokens:5}"
    )


def print_summary(results: list[Result]) -> None:
    """モデル×質問ごとに、成功した回の中央値をまとめる。"""
    print("\n=== 集計（成功した回の中央値） ===")
    print(
        f"{'model':24} {'q':7} {'ok':>5} {'TTFT':>6} {'応答':>6} "
        f"{'出力':>6} {'思考':>6} {'合計':>6} {'出力tok/s':>9}"
    )
    keys = dict.fromkeys((r.model, r.question) for r in results)
    for model, qid in keys:
        group = [r for r in results if r.model == model and r.question == qid]
        ok = [r for r in group if r.status == 200]
        if not ok:
            print(f"{model:24} {qid:7} {0:>2}/{len(group):<2}  （すべて失敗）")
            continue

        def med(attr: str) -> float:
            return statistics.median(getattr(r, attr) for r in ok)

        # 思考トークンも生成には時間がかかるので、出力+思考 を応答時間で割る
        tps = statistics.median((r.output_tokens + r.thoughts_tokens) / r.latency_s for r in ok)
        print(
            f"{model:24} {qid:7} {len(ok):>2}/{len(group):<2} {med('ttft_s'):6.2f} {med('latency_s'):6.2f} "
            f"{med('output_tokens'):6.0f} {med('thoughts_tokens'):6.0f} {med('total_tokens'):6.0f} {tps:9.1f}"
        )


def write_csv(results: list[Result], path: Path) -> None:
    rows = []
    for r in results:
        row = asdict(r)
        row.update(row.pop("quota"))
        rows.append(row)
    fields = list(asdict(Result("", "", 0)).keys())
    fields.remove("quota")
    fields += ["quota_id", "quota_value", "retry_delay"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV を保存しました: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini の応答時間とトークン数を計測する")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--questions", nargs="+", default=list(QUESTIONS), choices=list(QUESTIONS))
    parser.add_argument("--repeat", type=int, default=3, help="各組み合わせの実行回数")
    parser.add_argument("--interval", type=float, default=4.0, help="リクエスト間の待ち秒数（RPM対策）")
    parser.add_argument("--burst", type=int, default=0, help="指定回数を間隔0で連打する（レート制限の確認用）")
    parser.add_argument("--parallel", type=int, default=1, help="--burst のときに同時に送る数")
    parser.add_argument("--out", type=Path, help="1回ごとの結果を保存するCSVのパス")
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    results: list[Result] = []

    if args.burst:
        # 最も短い質問を連打して、何回目で 429 になるかを見る
        for model in args.models:
            print(f"[burst] {model} に {args.burst} 回連続で送信（同時 {args.parallel}）")
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                runs = pool.map(lambda i: measure(client, model, "short", i), range(1, args.burst + 1))
                for r in runs:
                    results.append(r)
                    print_line(r)
    else:
        for model in args.models:
            print(f"[model] {model}")
            for qid in args.questions:
                for i in range(1, args.repeat + 1):
                    r = measure(client, model, qid, i)
                    results.append(r)
                    print_line(r)
                    time.sleep(args.interval)

    print_summary(results)
    if args.out:
        write_csv(results, args.out)


if __name__ == "__main__":
    main()
