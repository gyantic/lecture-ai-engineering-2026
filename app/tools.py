"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import random
import re
from datetime import datetime
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RESULTS = 3
# 学習ログの保存先（.gitignore 済み。個人の記録なので共有しない）
LOG_DIR = DATA_DIR.parent / "learning_log"


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> dict:
    """指定した回（1始まりの整数）で扱う用語から1つ選び、「○○の意味は？」という小テストを作る。

    「第2回の復習問題を出して」「クイズを出して」のように、学習者が理解度を
    確かめたいときに使う。返り値は {"question": 問題文, "answer": 模範解答} の辞書。
    出題するときは question だけを見せ、answer は学習者が答えた後の答え合わせに使うこと。
    """
    terms = [e for e in load_glossary() if e["lecture"] == lecture]
    if not terms:
        available = sorted({e["lecture"] for e in load_glossary()})
        return {
            "question": "",
            "answer": f"第{lecture}回の用語は用語集にありません。用語がある回: {', '.join(map(str, available))}",
        }
    entry = random.choice(terms)
    return {"question": f"「{entry['term']}」の意味は？", "answer": entry["definition"]}


def save_learning_log(title: str, content: str) -> str:
    """学習の記録を learning_log/ フォルダに Markdown ファイルとして保存する。

    利用者が「今日学んだことを記録して」「この内容をログに残して」のように、
    学習の記録を残したいと頼んだときだけ使う。頼まれていないのに勝手に保存しない。
    title には記録の短い見出し（例: "第3回 エージェントとツール"）、
    content には保存する本文（学んだこと・疑問点などを Markdown で）を入れる。
    ファイルが作られるので、実行前に利用者の承認が必要になる。
    """
    # 見出しはファイル名にも使うので、パス区切りなどファイル名に使えない文字を除く
    safe_title = re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("._")[:50] or "log"
    now = datetime.now()
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{now:%Y%m%d-%H%M%S}_{safe_title}.md"
    # "x" モード: 同名のファイルがあれば上書きせずにエラーにする
    with open(path, "x", encoding="utf-8") as f:
        f.write(f"# {title}\n\n- 日時: {now:%Y-%m-%d %H:%M}\n\n{content.strip()}\n")
    return f"学習ログを保存しました: learning_log/{path.name}"
