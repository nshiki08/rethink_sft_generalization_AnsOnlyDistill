# @title 0. 認証（最初のセル、Colab のカーネルで実行）: Colab Secrets / 環境変数からトークンを読む。値は表示しない
# 読んだトークンは環境変数に入れ、セル 1 が起動する公式環境のカーネルへ引き継ぐ。HF へのログイン確認はセクション 2 で行う。
# 学習ログの写し先（AO_DRIVE_LOG_DIR、任意）もここで読み、Drive のフォルダなら Drive をマウントする。
import os
import sys


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

# ---- 学習ログの写し先（任意。トークンではないので値を表示する） ----------------------------------------------
# Colab は切断でランタイムのディスクが消える。Colab Secrets / 環境変数の AO_DRIVE_LOG_DIR に Drive のフォルダ（/content/drive/MyDrive/...）を
# 登録しておくと、学習中の loss の推移とログを別スレッドがそこへ写す。checkpoint は写さない（HF へ転送する）。
# 設定セルの LOG_MIRROR_DIR は写し先を変えるだけで、Drive をマウントしない。Drive をマウントするのはこのセル（AO_DRIVE_LOG_DIR が Drive のパスのとき）。
DRIVE_MOUNT_POINT = "/content/drive"
_log_dir = _get_secret("AO_DRIVE_LOG_DIR")
if not _log_dir:
    print("AO_DRIVE_LOG_DIR: なし。学習ログはランタイムのディスクにだけ残る（切断で消える）。登録した場合はこのセルを実行し直す")
else:
    _log_dir = os.path.expanduser(_log_dir)
    if not os.path.isabs(_log_dir):
        # 相対パスは直さずに渡す（Colab のカーネルと公式環境のカーネルは作業フォルダが違い、別の場所を指す）。写す側が「絶対パスでない」と断り、写さない
        print(f"WARN: AO_DRIVE_LOG_DIR={_log_dir!r} は絶対パスでない（例 {DRIVE_MOUNT_POINT}/MyDrive/<フォルダ>）。学習ログは写されない。"
              "値を直してこのセルを実行し直す")
    else:
        _log_dir = os.path.normpath(_log_dir)
        print("AO_DRIVE_LOG_DIR:", _log_dir)
        _on_drive = (_log_dir + os.sep).startswith(DRIVE_MOUNT_POINT + os.sep)
        if _on_drive and not os.path.ismount(DRIVE_MOUNT_POINT):
            try:
                from google.colab import drive  # type: ignore

                drive.mount(DRIVE_MOUNT_POINT)
            except Exception as _e:   # Colab の外、またはマウントを許可しなかった。学習は続けられる（写しだけ行われない）
                print(f"WARN: Drive を {DRIVE_MOUNT_POINT} にマウントできない（{type(_e).__name__}: {str(_e)[:200]}）。学習ログは写されない")
        elif not _on_drive and "google.colab" in sys.modules:   # Colab のカーネルは起動時に google.colab を読み込んでいる
            print(f"WARN: {_log_dir} は Drive（{DRIVE_MOUNT_POINT}）の外で、ランタイムのディスクにある。切断で消える")
    os.environ["AO_DRIVE_LOG_DIR"] = _log_dir
    # セル 1 の後にこのセルを実行し直した場合: 公式環境のカーネルの環境変数は起動時の写しなので、動いているカーネルにも渡す
    _k = globals().get("AO_KERNEL")
    if _k is not None:
        try:
            _k.run("import os; os.environ['AO_DRIVE_LOG_DIR'] = " + repr(_log_dir))
            print("  動いている公式環境のカーネルにも渡した（次の学習の起動から写す）")
        except Exception as _e:
            print(f"WARN: 公式環境のカーネルに AO_DRIVE_LOG_DIR を渡せない（{type(_e).__name__}: {str(_e)[:200]}）。セル 1 から実行し直す")
