"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールを持たない「素の」呼び出し。会話履歴は保持し、続けて質問できる。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


EXIT_WORDS = {"exit", "quit", ""}


def ask(chat, question: str) -> str:
    # chat が会話履歴を保持しているので、呼ぶたびに履歴が積み上がる
    response = chat.send_message(question)

    usage = response.usage_metadata
    print(
        f"[トークン] 入力: {usage.prompt_token_count} / "
        f"出力: {usage.candidates_token_count}"
    )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)

    question = args.question
    print("（終了するには exit / quit / 空行 / Ctrl-D）")
    while True:
        print(f"[質問] {question}\n")
        try:
            print(ask(chat, question))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)

        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if question.lower() in EXIT_WORDS:
            break
        print()


if __name__ == "__main__":
    main()
