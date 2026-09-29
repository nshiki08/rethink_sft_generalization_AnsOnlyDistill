# @title 0. 認証（最初のセル、Colab のカーネルで実行）: Colab Secrets / 環境変数からトークンを読む。値は表示しない
# 読んだトークンは環境変数に入れ、セル 1 が起動する公式環境のカーネルへ引き継ぐ。HF へのログイン確認はセクション 2 で行う。
import os


def _get_secret(name):
    value = None
    try:
        from google.colab import userdata  # type: ignore

        try:
            value = userdata.get(name)
        except Exception:
            value = None
    except ImportError:
        pass
    if not value:
        value = os.environ.get(name)
    return value or None


for _name, _note in (("HF_TOKEN", "HF へのアップロード・再開は無効。Colab Secrets に HF_TOKEN（write）を登録して再実行する"),
                     ("WANDB_API_KEY", "公式スクリプトと同じ offline mode を使う")):
    _v = _get_secret(_name)
    if _v:
        os.environ[_name] = _v
        print(f"{_name}: found (masked)")
    else:
        print(f"{_name}: なし。{_note}")
del _v
