# @title 0. 認証（最初のセル）: Colab Secrets / 環境変数からトークンを読む。値は表示しない
import os, sys, subprocess


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


_hf_token = _get_secret("HF_TOKEN")
_wandb_key = _get_secret("WANDB_API_KEY")

HF_LOGGED_IN = False
HF_ACCOUNT_NAME = None  # HF の whoami から取る。GitHub のユーザー名からは推測しない

# huggingface_hub はこのカーネルで最初に import する前に、公式 requirements.txt と同じ版へ揃える（import 後の pip では差し替わらない）
HUGGINGFACE_HUB_PIN = "0.34.4"
try:
    from importlib.metadata import version as _pkg_version
    _hub_installed = _pkg_version("huggingface_hub")
except Exception:
    _hub_installed = None
if _hub_installed != HUGGINGFACE_HUB_PIN:
    assert "huggingface_hub" not in sys.modules, "huggingface_hub が既に import されている。ランタイムを再起動してこのセルから実行する"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"huggingface_hub=={HUGGINGFACE_HUB_PIN}"])
import huggingface_hub
print("huggingface_hub", huggingface_hub.__version__)

if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
    huggingface_hub.login(token=_hf_token, add_to_git_credential=False)
    _who = huggingface_hub.whoami()
    HF_ACCOUNT_NAME = _who.get("name")
    HF_LOGGED_IN = True
    print(f"HF: logged in as '{HF_ACCOUNT_NAME}' (token masked)")
else:
    print("HF: HF_TOKEN が見つからない。HF へのアップロード・再開は無効。Colab Secrets に HF_TOKEN を登録して再実行する")

if _wandb_key:
    os.environ["WANDB_API_KEY"] = _wandb_key
    print("WANDB: API key found (masked)")
else:
    print("WANDB: API key なし。公式スクリプトと同じ offline mode を使う")

del _hf_token, _wandb_key
