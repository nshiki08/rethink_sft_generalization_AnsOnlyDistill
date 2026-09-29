# @title 6-a. 起動・監視・HF 転送・再開のヘルパー（追加処理はすべてここ。公式 trainer は subprocess で無変更のまま実行）
import subprocess, threading, time, json, os, re, glob, shutil, zipfile, hashlib, random, datetime, io, sys, signal, fnmatch
import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

HF_API = HfApi() if HF_LOGGED_IN else None


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def require_training_env(n_gpus=None):
    """学習（試走・本学習・再開）を始める前の環境条件: GPU・flash-attn・1 GPU の FSDP2 勾配確認"""
    n_gpus = n_gpus or N_GPUS
    assert N_GPUS >= 1 and FLASH_ATTN_OK, "GPU と flash-attn が必要"
    if REQUIRE_FSDP2_GRAD_CHECK and n_gpus == 1:
        assert FSDP2_GRAD_CHECK.get("status") == "ok", (
            f"1 GPU の FSDP2 勾配確認（セクション 2）が OK ではない: {FSDP2_GRAD_CHECK.get('status')}。"
            "world size 1 の NCCL AVG で勾配が壊れる既知の報告があるため学習を開始しない")


def hash_file(path, with_sha256=True):
    """1 回の読み込みで git blob sha1（小さいファイルの照合用）と sha256（LFS 照合用）を計算する"""
    size = os.path.getsize(path)
    h1 = hashlib.sha1(f"blob {size}\0".encode())
    h2 = hashlib.sha256() if with_sha256 else None
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h1.update(b)
            if h2 is not None:
                h2.update(b)
    return h1.hexdigest(), (h2.hexdigest() if h2 is not None else None)


def git_blob_sha1(path):
    return hash_file(path, with_sha256=False)[0]


# ---- checkpoint の完全性 -----------------------------------------------------------------------
OPTIM_FILE_PATTERN = "optim_world_size_*"


def checkpoint_expected_files(world_size, content="full"):
    """content="model_only" は HF へ重みだけ送った checkpoint（optimizer 状態なし。分析・変換用で再開には使えない）"""
    files = []
    for r in range(world_size):
        files += [f"model_world_size_{world_size}_rank_{r}.pt", f"extra_state_world_size_{world_size}_rank_{r}.pt"]
        if content == "full":
            files.append(f"optim_world_size_{world_size}_rank_{r}.pt")
    files += ["fsdp_config.json", "data.pt", "huggingface/config.json", "huggingface/tokenizer_config.json"]
    return files


def checkpoint_is_complete(ckpt_dir, world_size, prev_sizes=None, require_stable=True, content="full"):
    """必要ファイルの存在・非空・torch.save(zip) の完全性・tracker・（連続 2 回のスキャンで）サイズ不変 を確認する。
    ディレクトリの存在や latest tracker だけでは完了と判断しない。"""
    step = int(re.search(r"global_step_(\d+)", ckpt_dir).group(1))
    sizes = {}
    for rel in checkpoint_expected_files(world_size, content):
        p = os.path.join(ckpt_dir, rel)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return False, sizes
        sizes[rel] = os.path.getsize(p)
        if rel.endswith(".pt") and not zipfile.is_zipfile(p):   # torch.save は zip 形式。末尾まで書けていれば central directory がある
            return False, sizes
    tokenizer_ok = any(os.path.isfile(os.path.join(ckpt_dir, "huggingface", f)) for f in ("tokenizer.json", "vocab.json", "tokenizer.model"))
    if not tokenizer_ok:
        return False, sizes
    tracker = os.path.join(os.path.dirname(ckpt_dir), "latest_checkpointed_iteration.txt")
    try:
        if int(open(tracker).read().strip()) < step:
            return False, sizes
    except Exception:
        return False, sizes
    for root, _, fs in os.walk(ckpt_dir):
        for f in fs:
            rel = os.path.relpath(os.path.join(root, f), ckpt_dir)
            sizes[rel] = os.path.getsize(os.path.join(root, f))
    if require_stable and prev_sizes != sizes:
        return False, sizes
    return True, sizes


def checkpoint_content_for(spec, step):
    """HF へ送る内容: 再開用 step（upload_full_steps）は optimizer 状態込み、それ以外は重みのみ。旧版の spec は全 step を full とする"""
    full = spec.get("upload_full_steps")
    return "full" if full is None or step in full else "model_only"


def build_manifest(spec, step, ckpt_dir, with_sha256=None, content="full"):
    with_sha256 = CKPT_MANIFEST_SHA256 if with_sha256 is None else with_sha256
    files = []
    for root, _, fs in os.walk(ckpt_dir):
        for f in sorted(fs):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, ckpt_dir)
            if rel in ("ao_ckpt_manifest.json", "ao_upload_verified.json"):
                continue
            if content == "model_only" and fnmatch.fnmatch(rel, OPTIM_FILE_PATTERN):
                continue
            sha1, sha256 = hash_file(p, with_sha256=with_sha256)
            ent = dict(path=rel, size=os.path.getsize(p), git_blob_sha1=sha1)
            if sha256:
                ent["sha256"] = sha256
            files.append(ent)
    manifest = dict(run_id=spec["run_id"], step=step, world_size=spec["n_gpus"], content=content, files=files, total_bytes=sum(f["size"] for f in files),
                    run_config=spec, created_at=now_iso(), fork_commit=FORK_COMMIT, data_sha256=spec["data_sha256"], base_revision=spec["base_revision"])
    with open(os.path.join(ckpt_dir, "ao_ckpt_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


# ---- HF 転送 -----------------------------------------------------------------------------------
def ensure_ckpt_repo():
    assert HF_API is not None, "HF にログインしていない（最初のセルで HF_TOKEN を設定）"
    assert HF_CKPT_REPO_ID, "HF_CKPT_REPO_ID が未設定"
    HF_API.create_repo(HF_CKPT_REPO_ID, private=HF_CKPT_PRIVATE, repo_type="model", exist_ok=True)


def hf_verify_uploaded(repo_id, path_in_repo, manifest, revision):
    """転送後に HF 側のサイズ・LFS sha256・git blob sha1 を manifest と照合する"""
    paths = [f"{path_in_repo}/{e['path']}" for e in manifest["files"]] + [f"{path_in_repo}/ao_ckpt_manifest.json"]
    infos = {i.path: i for i in HF_API.get_paths_info(repo_id, paths=paths, revision=revision, repo_type="model")}
    mismatches = []
    for e in manifest["files"]:
        rp = f"{path_in_repo}/{e['path']}"
        i = infos.get(rp)
        if i is None:
            mismatches.append((e["path"], "missing on HF")); continue
        if getattr(i, "size", None) != e["size"]:
            mismatches.append((e["path"], f"size {getattr(i, 'size', None)} != {e['size']}")); continue
        lfs = getattr(i, "lfs", None)
        if lfs is not None and e.get("sha256") and getattr(lfs, "sha256", None) and lfs.sha256 != e["sha256"]:
            mismatches.append((e["path"], "sha256 mismatch"))
        elif lfs is None and getattr(i, "blob_id", None) and i.blob_id != e["git_blob_sha1"]:
            mismatches.append((e["path"], "git blob sha1 mismatch"))
    if f"{path_in_repo}/ao_ckpt_manifest.json" not in infos:
        mismatches.append(("ao_ckpt_manifest.json", "missing on HF"))
    return mismatches


def upload_checkpoint(spec, step, ckpt_dir, log=print):
    """保存完了した checkpoint を runs/<run_id>/global_step_<step>/ へ転送し、検証後に完了マーカーを置く。既存の完了済み step は上書きしない"""
    t0 = time.time()
    ensure_ckpt_repo()
    pir = f"runs/{spec['run_id']}/global_step_{step}"
    remote = hf_run_paths(HF_CKPT_REPO_ID, spec["run_id"]).get(step)
    if remote and remote["complete"]:
        log(f"collision: {pir} は HF に完了済みで存在する。上書きしない")
        return dict(status="collision_remote_complete", path_in_repo=pir)
    if remote:
        log(f"{pir} に未完了の転送が残っている。同じ step を再転送する")
    content = checkpoint_content_for(spec, step)
    manifest = build_manifest(spec, step, ckpt_dir, content=content)
    ignore = ["ao_upload_verified.json"] + ([OPTIM_FILE_PATTERN] if content == "model_only" else [])
    last_err = None
    for attempt in range(5):
        try:
            info = HF_API.upload_folder(folder_path=ckpt_dir, path_in_repo=pir, repo_id=HF_CKPT_REPO_ID, repo_type="model",
                                        commit_message=f"{spec['run_id']} global_step_{step} ({content})", ignore_patterns=ignore)
            break
        except Exception as e:  # noqa
            last_err = e
            wait = 2 ** attempt * 5
            log(f"upload attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:200]} -> retry in {wait}s")
            time.sleep(wait)
    else:
        return dict(status="upload_failed", error=repr(last_err)[:500], path_in_repo=pir, duration_sec=time.time() - t0)
    mism = hf_verify_uploaded(HF_CKPT_REPO_ID, pir, manifest, info.oid)
    if mism:
        log(f"verify failed for {pir}: {mism[:5]}")
        return dict(status="verify_failed", commit=info.oid, mismatches=mism, path_in_repo=pir, duration_sec=time.time() - t0)
    marker = dict(run_id=spec["run_id"], step=step, content=content, folder_commit=info.oid, verified_at=now_iso(), n_files=len(manifest["files"]),
                  total_bytes=manifest["total_bytes"], world_size=spec["n_gpus"], total_steps=spec["total_steps"])
    with open(os.path.join(ckpt_dir, "ao_upload_verified.json"), "w") as f:
        json.dump(marker, f, indent=2)
    info2 = HF_API.upload_file(path_or_fileobj=os.path.join(ckpt_dir, "ao_upload_verified.json"), path_in_repo=f"{pir}/ao_upload_verified.json",
                               repo_id=HF_CKPT_REPO_ID, repo_type="model", commit_message=f"{spec['run_id']} global_step_{step} verified")
    dur = time.time() - t0
    log(f"uploaded+verified {pir} [{content}] ({manifest['total_bytes'] / 1e9:.2f} GB, {dur:.0f}s) commit={info.oid[:10]} marker={info2.oid[:10]}")
    return dict(status="verified", content=content, commit=info.oid, marker_commit=info2.oid, path_in_repo=pir, total_bytes=manifest["total_bytes"], duration_sec=dur,
                url=f"https://huggingface.co/{HF_CKPT_REPO_ID}/tree/{info2.oid}/{pir}")


# ---- 監視スレッド ------------------------------------------------------------------------------
class TrainingMonitor(threading.Thread):
    """学習プロセスと並行して checkpoint ディレクトリを監視し、保存完了ごとに HF へ転送・検証する。GPU メモリも記録する。
    kill_after_step が指定された場合（試走）: その step の保存完了を検知したら学習プロセス群を SIGSTOP で一時停止し、
    HF 転送・検証が終わってから SIGTERM で終了する（保存 → HF 転送 → プロセス終了 の順を保証する）。"""

    def __init__(self, proc, spec, kill_after_step=None, upload=True, poll_sec=10, keep_local=None, stable_scans=True):
        super().__init__(daemon=True)
        self.proc, self.spec, self.kill_after_step, self.upload, self.poll_sec = proc, spec, kill_after_step, upload, poll_sec
        self.keep_local = LOCAL_KEEP_LAST_N_VERIFIED_CKPTS if keep_local is None else keep_local
        self.stable_scans = stable_scans   # True: 連続 2 回のスキャンでサイズ不変を要求（通常）。試走では tracker + zip 検証のみで即時判定
        self.ckpt_root = spec["overrides"]["trainer.default_local_dir"]
        self.state = dict(uploaded={}, failed={}, complete_local=[], gpu_mem_max_mb={}, events=[], deleted_local=[], skipped=[])
        self._sizes, self._attempts, self.killed, self.paused = {}, {}, False, False

    def _signal_group(self, sig):
        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
            return True
        except ProcessLookupError:
            return False

    def _terminate(self):
        alive = self.proc.poll() is None
        if not alive:
            self.log("プロセスは既に終了していた（kill は不要）")
            return False
        if self.paused:
            self._signal_group(signal.SIGCONT)
            self.paused = False
        self._signal_group(signal.SIGTERM)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._signal_group(signal.SIGKILL)
        self.killed = True
        return True

    def log(self, msg):
        line = f"[monitor {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.state["events"].append(dict(t=time.time(), msg=msg))

    def _sample_gpu(self):
        if shutil.which("nvidia-smi") is None:
            return
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            try:
                i, used = [x.strip() for x in line.split(",")]
                self.state["gpu_mem_max_mb"][i] = max(self.state["gpu_mem_max_mb"].get(i, 0), int(float(used)))
            except ValueError:
                pass

    def _cleanup_local(self):
        verified = sorted(int(s) for s in self.state["uploaded"] if self.state["uploaded"][s]["status"] == "verified")
        resume_src = self.spec.get("resume_from_path")
        for s in verified[: max(0, len(verified) - self.keep_local)]:
            d = os.path.join(self.ckpt_root, f"global_step_{s}")
            if resume_src and os.path.abspath(d) == os.path.abspath(resume_src):
                continue   # 再開元は消さない
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
                self.state["deleted_local"].append(s)
                self.log(f"local cleanup: removed global_step_{s} (HF 転送・検証済み)")

    def _scan(self, final=False):
        dirs = glob.glob(os.path.join(self.ckpt_root, "global_step_*"))
        steps = sorted(int(re.search(r"global_step_(\d+)$", d).group(1)) for d in dirs if re.search(r"global_step_(\d+)$", d))
        for step in steps:
            if step in self.state["uploaded"]:
                continue
            d = os.path.join(self.ckpt_root, f"global_step_{step}")
            marker = os.path.join(d, "ao_upload_verified.json")
            if os.path.isfile(marker):   # 以前の実行で転送・検証済み（再開のために HF から取得したもの）。再転送しない
                try:
                    self.state["uploaded"][step] = dict(status="verified", previously_verified=True, **json.load(open(marker)))
                    self.log(f"global_step_{step} は転送・検証済み（マーカーあり）。スキップ")
                except Exception:
                    pass
                continue
            complete, sizes = checkpoint_is_complete(d, self.spec["n_gpus"], self._sizes.get(step), require_stable=(self.stable_scans and not final))
            self._sizes[step] = sizes
            if not complete:
                continue
            if step not in self.state["complete_local"]:
                self.state["complete_local"].append(step)
                self.log(f"checkpoint global_step_{step} complete locally ({sum(sizes.values()) / 1e9:.2f} GB, {len(sizes)} files)")
            kill_now = self.kill_after_step is not None and step >= self.kill_after_step and not self.killed
            upload_steps = self.spec.get("upload_steps")
            if upload_steps is not None and step not in upload_steps and step not in self.state["skipped"]:
                # 転送対象外の step（公開 CoT 学生と同じ step でも再開用の間隔でもない）。HF に無いので再開候補にならず、ローカルからも消す
                self.state["skipped"].append(step)
                shutil.rmtree(d, ignore_errors=True)
                self.log(f"global_step_{step} は転送対象外（HF_UPLOAD_STEPS={self.spec.get('upload_policy')}）。ローカルから削除")
                continue
            if step in self.state["skipped"]:
                continue
            if kill_now and not self.paused and self.proc.poll() is None:
                self.paused = self._signal_group(signal.SIGSTOP)   # 転送中に学習が先へ進まないよう一時停止（切断前の状態を固定）
                self.log(f"kill_after_step={self.kill_after_step}: global_step_{step} の保存完了を検知。学習プロセスを一時停止して HF 転送する")
            if self.upload:
                if self._attempts.get(step, 0) >= 3:
                    continue
                self._attempts[step] = self._attempts.get(step, 0) + 1
                res = upload_checkpoint(self.spec, step, d, log=self.log)
                if res["status"] == "verified":
                    self.state["uploaded"][step] = res
                    self.state["failed"].pop(step, None)
                    self._cleanup_local()
                elif res["status"].startswith("collision"):
                    # 同じ step が HF に完了済み（例: 再開元）。上書きしない。転送済みとして扱う
                    self.state["uploaded"][step] = dict(res, status="verified", previously_verified=True)
                    self._attempts[step] = 99
                else:
                    self.state["failed"][step] = res
            if kill_now and (not self.upload or step in self.state["uploaded"]):
                self.log(f"global_step_{step} の{'HF 転送・検証' if self.upload else '保存'}を確認したのでプロセスを終了する（切断の模擬）")
                self._terminate()

    def run(self):
        while True:
            alive = self.proc.poll() is None
            self._sample_gpu()
            try:
                self._scan(final=not alive)
            except Exception as e:  # noqa
                self.log(f"monitor error: {type(e).__name__}: {str(e)[:300]}")
                if self.paused:   # 転送に失敗しても停止したまま放置しない
                    self._terminate()
            if not alive:
                try:
                    self._scan(final=True)
                except Exception as e:  # noqa
                    self.log(f"monitor error: {type(e).__name__}: {str(e)[:300]}")
                break
            time.sleep(self.poll_sec)


# ---- 起動 --------------------------------------------------------------------------------------
_METRIC_RE = re.compile(r"(?:^|\s)step:(\d+)((?: - [\w/]+:\S+)+)")


def parse_metric_line(line):
    """console logger の `step:N - train/loss:0.5 - train/lr:1e-05 ...` を辞書にする（nan/inf も float として保持）"""
    m = _METRIC_RE.search(line)
    if not m:
        return None
    d = dict(step=int(m.group(1)))
    for kv in m.group(2).split(" - ")[1:]:
        k, v = kv.split(":", 1)
        try:
            d[k] = float(v)
        except ValueError:
            pass
    return d


_INTERESTING = ("Saving checkpoint", "Updated checkpoint tracker", "Loaded", "loaded", "resume", "Traceback", "Error", "error", "OutOfMemory", "Total training steps",
                "Number of steps/epoch", "Normalize batch", "Using FSDP rank", "WARN", "Killed", "Signal", "Found checkpoint", "grad_norm is not finite")


def launch_training(spec, kill_after_step=None, upload=None, keep_local=None, quiet=False, poll_sec=10, stable_scans=True):
    """公式エントリーポイントを torchrun (python -m torch.distributed.run) で起動し、ログを流しつつ監視スレッドで checkpoint を HF へ転送する。終了まで戻らない。"""
    upload = HF_UPLOAD_CHECKPOINTS if upload is None else upload
    if upload:
        ensure_ckpt_repo()
    run_id = spec["run_id"]
    run_log_dir = f"{LOG_DIR}/{run_id}"
    os.makedirs(run_log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = f"{run_log_dir}/{ts}.log"
    with open(f"{run_log_dir}/{ts}.spec.json", "w") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    port = random.randint(20000, 21000)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={spec['n_gpus']}", "--node_rank=0", "--master_addr=127.0.0.1",
           f"--master_port={port}", "-m", spec["module"]] + [f"{k}={v}" for k, v in spec["overrides"].items()]
    env = os.environ.copy()
    env.update(spec["env"])
    env["PYTHONPATH"] = REPO_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    with open(f"{run_log_dir}/{ts}.cmd.txt", "w") as f:
        f.write(" \\\n  ".join(cmd) + "\n\nenv: " + json.dumps(spec["env"]) + "\n")
    print(f"launch {run_id}: log={log_path}")
    print("  " + " ".join(cmd[:10]) + " ...")
    t_launch = time.time()
    # start_new_session=True: torchrun とワーカーを 1 つのプロセスグループにし、試走の一時停止/終了をグループごと送れるようにする
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace",
                            start_new_session=True)
    with open(f"{WORK_DIR}/ao-trainer.pgid", "w") as f:   # セル 1 を再実行したとき、残った学習プロセスをこの process group ごと終了する
        f.write(str(proc.pid))
    mon = TrainingMonitor(proc, spec, kill_after_step=kill_after_step, upload=upload, keep_local=keep_local, poll_sec=poll_sec, stable_scans=stable_scans)
    mon.start()
    metrics, events = [], []
    try:
        with open(log_path, "w") as lf:
            for line in proc.stdout:
                t = time.time()
                lf.write(f"[{t - t_launch:9.1f}s] {line}")
                m = parse_metric_line(line)
                if m:
                    m["t"] = t
                    _scale = float(spec.get("logged_loss_scale", 1.0))
                    if "train/loss" in m:
                        m["train/loss_official_scale"] = m["train/loss"] / _scale   # adv-only で N/8 倍されたログを公式スケールへ戻した値
                    metrics.append(m)
                    if not quiet:
                        print(f"[{t - t_launch:7.0f}s] step {m['step']}/{spec['total_steps']} loss={m.get('train/loss_official_scale', float('nan')):.4f}"
                              f"{'' if _scale == 1.0 else f' (logged {_scale:g}x: ' + format(m.get('train/loss', float('nan')), '.4f') + ')'} "
                              f"lr={m.get('train/lr', float('nan')):.3e} grad_norm={m.get('train/grad_norm', float('nan')):.3f} entropy={m.get('train/entropy/mean', float('nan')):.3f}", flush=True)
                elif any(k in line for k in _INTERESTING) and "ratio_list_all" not in line:
                    events.append(dict(t=t, line=line.rstrip()[:300]))
                    if not quiet:
                        print(f"[{t - t_launch:7.0f}s] {line.rstrip()[:200]}", flush=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt -> terminating training process group")
        mon._terminate()
    rc = proc.wait()
    try:
        mon.join(timeout=3600 * 6)   # プロセス終了後も、最後の checkpoint の転送が終わるまで待つ
    except KeyboardInterrupt:
        print("KeyboardInterrupt during final upload -> 結果は転送未完了のまま記録する")
    t_end = time.time()
    result = dict(run_id=run_id, spec=spec, exit_code=rc, killed_by_monitor=mon.killed, log_path=log_path, launched_at=t_launch, ended_at=t_end,
                  wall_sec=t_end - t_launch, metrics=metrics, events=events, monitor=dict(uploaded=mon.state["uploaded"], failed=mon.state["failed"],
                  complete_local=mon.state["complete_local"], gpu_mem_max_mb=mon.state["gpu_mem_max_mb"], deleted_local=mon.state["deleted_local"], skipped=mon.state["skipped"],
                  events=mon.state["events"]))
    with open(f"{run_log_dir}/{ts}.result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    status = "killed(simulated disconnect)" if mon.killed else ("OK" if rc == 0 else f"FAILED rc={rc}")
    print(f"finished {run_id}: {status} | steps logged={len(metrics)} | wall={t_end - t_launch:.0f}s | GPU mem max MB={mon.state['gpu_mem_max_mb']} "
          f"| uploaded={sorted(mon.state['uploaded'])} | failed={ {k: v['status'] for k, v in mon.state['failed'].items()} }")
    if mon.state["failed"]:
        print("*** HF 転送に失敗した checkpoint がある。ローカルには残している。ネットワーク復旧後に upload_checkpoint() を再実行する ***")
    return result


def summarize_timing(result):
    """初期化時間・optimizer step 時間（勾配蓄積込み）・保存時間・転送時間を分けて集計する"""
    ms = result["metrics"]
    spec = result["spec"]
    out = dict(n_steps=len(ms))
    if not ms:
        return out
    saves = set(spec["expected_save_steps"])
    intervals = [(ms[i]["step"], ms[i]["t"] - ms[i - 1]["t"]) for i in range(1, len(ms))]
    plain = [d for s, d in intervals if (s - 1) not in saves]      # 直前 step で保存が走った区間を除く
    out["step_sec_median"] = float(np.median(plain)) if plain else None
    out["step_sec_all"] = [round(d, 2) for _, d in intervals]
    first_step_t = ms[0]["t"]
    out["init_sec_incl_first_step"] = first_step_t - result["launched_at"]
    out["init_sec_est"] = (first_step_t - result["launched_at"] - out["step_sec_median"]) if out["step_sec_median"] else None
    ev = result["events"]
    save_starts = [e["t"] for e in ev if "Saving checkpoint to" in e["line"]]
    save_ends = [e["t"] for e in ev if "Updated checkpoint tracker" in e["line"]]
    out["save_sec"] = [round(b - a, 1) for a, b in zip(save_starts, save_ends)]
    out["upload_sec"] = {s: round(v.get("duration_sec", 0), 1) for s, v in result["monitor"]["uploaded"].items()}
    out["gpu_mem_max_mb"] = result["monitor"]["gpu_mem_max_mb"]
    return out


# ---- HF からの再開 ------------------------------------------------------------------------------
def list_hf_checkpoints(run_id):
    """HF 上の run の checkpoint 一覧。完了マーカーのあるものだけを再開候補にする"""
    paths = hf_run_paths(HF_CKPT_REPO_ID, run_id)
    rows = []
    for step in sorted(paths):
        row = dict(step=step, complete=paths[step]["complete"], n_files=len(paths[step]["files"]))
        if row["complete"]:
            try:
                mp = hf_hub_download(HF_CKPT_REPO_ID, f"runs/{run_id}/global_step_{step}/ao_upload_verified.json", repo_type="model", local_dir=f"{WORK_DIR}/hf_markers")
                row.update(json.load(open(mp)))
            except Exception as e:  # noqa
                row["complete"] = False
                row["marker_error"] = str(e)[:200]
        row.setdefault("content", "full")   # 旧版のマーカーには content が無い（全ファイルを送っていた）
        row["resumable"] = row["complete"] and row["content"] == "full"
        rows.append(row)
    return rows


def download_hf_checkpoint(run_id, step, revision=None):
    """HF から checkpoint を取得し manifest で検証する。戻り値: (local_path, manifest, revision)"""
    pir = f"runs/{run_id}/global_step_{step}"
    marker_path = hf_hub_download(HF_CKPT_REPO_ID, f"{pir}/ao_upload_verified.json", repo_type="model", local_dir=f"{WORK_DIR}/hf_markers")
    marker = json.load(open(marker_path))
    revision = revision or marker["folder_commit"]
    local = f"{CKPT_DIR}/{run_id}/global_step_{step}"
    tmp = f"{WORK_DIR}/hf_download/{run_id}_{step}"
    shutil.rmtree(tmp, ignore_errors=True)
    snapshot_download(HF_CKPT_REPO_ID, repo_type="model", revision=revision, allow_patterns=[f"{pir}/*", f"{pir}/**"], local_dir=tmp)
    src = os.path.join(tmp, pir)
    if os.path.isdir(local):
        shutil.rmtree(local)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    shutil.move(src, local)
    shutil.copy(marker_path, os.path.join(local, "ao_upload_verified.json"))   # マーカーは folder_commit より後の commit なので別途置く（監視が再転送しないための目印）
    manifest = json.load(open(os.path.join(local, "ao_ckpt_manifest.json")))
    bad = []
    for e in manifest["files"]:
        p = os.path.join(local, e["path"])
        if not os.path.isfile(p) or os.path.getsize(p) != e["size"]:
            bad.append((e["path"], "missing/size")); continue
        if e.get("sha256") and sha256_of(p) != e["sha256"]:
            bad.append((e["path"], "sha256"))
    if bad:
        raise RuntimeError(f"取得した checkpoint が manifest と一致しない: {bad[:5]}")
    # 公式の latest tracker も揃えておく（resume_path モードでは使われないが、ディレクトリ構成を学習時と同じにする）。
    # 既存の値より大きいときだけ書く（小さい値にすると、ローカルに残した新しい step の checkpoint が「未完了」と判定される）
    _tracker = os.path.join(os.path.dirname(local), "latest_checkpointed_iteration.txt")
    try:
        _prev = int(open(_tracker).read().strip())
    except (OSError, ValueError):
        _prev = -1
    if step > _prev:
        with open(_tracker, "w") as f:
            f.write(str(step))
    ok, _ = checkpoint_is_complete(local, manifest["world_size"], require_stable=False, content=manifest.get("content", "full"))
    assert ok, "必要ファイル（model/optim/extra の各 rank shard, data.pt, fsdp_config.json, huggingface/）が揃っていない"
    manifest["folder_commit"] = marker.get("folder_commit")
    print(f"downloaded {pir} [{manifest.get('content', 'full')}] @ {revision[:10]} -> {local} ({manifest['total_bytes'] / 1e9:.2f} GB, verified against manifest)")
    return local, manifest, revision


def current_data_sha256_for(saved_data_path):
    """保存時の data_path に対応する現在のファイルの sha256（本学習は AO parquet、試走は subset parquet）"""
    if os.path.basename(saved_data_path) == os.path.basename(AO_PARQUET):
        return AO_SHA256, AO_PARQUET
    if os.path.isfile(saved_data_path):
        return sha256_of(saved_data_path), saved_data_path
    return None, None


def saved_training_view(saved, world_size):
    """保存時の run 設定にある training_view。旧版（公式 8 GPU 再現を入れる前）の checkpoint には無いので、並べ替えなし・vanilla とみなす"""
    if saved.get("training_view"):
        return saved["training_view"]
    return dict(enabled=False, official_world=OFFICIAL_WORLD_SIZE, actual_world=int(world_size), advantage=1.0,
                importance_sampling_mode=saved["overrides"].get("trainer.importance_sampling_mode", "vanilla"), order_sha256=None)


def check_resume_compatibility(manifest, n_gpus=None):
    """コード・データ・tokenizer・学習設定・GPU 台数/FSDP 構成の互換性を確認し、問題を列挙する"""
    n_gpus = n_gpus or N_GPUS
    saved = manifest["run_config"]
    problems, warnings_ = [], []
    if manifest["world_size"] != n_gpus:
        problems.append(f"FSDP world_size {manifest['world_size']} != 現在の GPU 台数 {n_gpus}（sharded checkpoint は台数依存。条件を黙って変えない）")
    if saved["base_revision"] != MODEL_INFO["base_revision"] or saved["model_key"] != MODEL_KEY:
        problems.append(f"Base モデル/revision が異なる: saved={saved['model_key']}@{saved['base_revision'][:8]} now={MODEL_KEY}@{MODEL_INFO['base_revision'][:8]}")
    cur_sha, cur_path = current_data_sha256_for(saved["data_path"])
    if cur_sha is None:
        problems.append(f"学習データ {saved['data_path']} が現在の環境に無い（セクション 3 / 試走セルで再作成する）")
    elif saved["data_sha256"] != cur_sha:
        problems.append(f"学習データの sha256 が異なる: saved={saved['data_sha256'][:12]} now={cur_sha[:12]} ({cur_path})")
    if saved["target_style"] != AO_TARGET_STYLE:
        problems.append("AO_TARGET_STYLE が異なる")
    # 公式 8 GPU 再現用の学習 file（並べ替えと advantage）が保存時と同じになるか
    sv = saved_training_view(saved, manifest["world_size"])
    if not saved.get("training_view"):
        warnings_.append("保存時の設定に training_view が無い（公式 8 GPU 再現を入れる前の版の checkpoint）。並べ替えなし・vanilla として扱う。"
                         "再開するには EMULATE_OFFICIAL_WORLD_SIZE=False にする（その run は公式 8 GPU の条件と異なる）")
    if cur_sha is not None and saved["data_sha256"] == cur_sha:
        _, tsha, tv = prepare_train_file(cur_path, cur_sha, manifest["world_size"])
        keys = ("enabled", "official_world", "actual_world", "advantage", "importance_sampling_mode", "order_sha256")
        diff = {k: (sv.get(k), tv.get(k)) for k in keys if sv.get(k) != tv.get(k)}
        if diff:
            problems.append(f"公式 8 GPU 再現の設定が保存時と異なる（EMULATE_OFFICIAL_WORLD_SIZE / GPU 台数を保存時に合わせる）: {diff}")
        elif sv.get("enabled") and tsha != saved.get("train_file_sha256"):
            warnings_.append("学習用 file の sha256 が保存時と異なる（行順・advantage は同一。parquet の書き込みメタデータの差の可能性）")
    if saved.get("fork_commit") != FORK_COMMIT:
        if OFFICIAL_CODE_UNCHANGED is True:
            warnings_.append(f"Fork commit が異なる: saved={str(saved.get('fork_commit'))[:10]} now={FORK_COMMIT[:10]}（verl/・training_scripts/ は参照 commit と同一なので続行可）")
        else:
            problems.append(f"Fork commit が異なり、公式コードの無変更を確認できない (OFFICIAL_CODE_UNCHANGED={OFFICIAL_CODE_UNCHANGED})")
    if saved["kind"] != "trial" and "trainer.total_training_steps" in saved["overrides"]:
        problems.append("保存時の設定に step 上限 (trainer.total_training_steps) がある。本学習の checkpoint ではない")
    if saved.get("max_length") != int(MAX_LENGTH):
        warnings_.append(f"現在の MAX_LENGTH {MAX_LENGTH} と保存時 {saved.get('max_length')} が異なる。再開は保存時の設定を使う")
    if not FLASH_ATTN_OK and n_gpus > 0:
        problems.append("flash-attn が使えない")
    if manifest.get("content", "full") != "full":
        problems.append("この checkpoint は重みのみ（optimizer 状態なし）。再開には使えない")
    if saved.get("gpu_profile", "official") != GPU_PROFILE:
        problems.append(f"GPU プロファイル（torch の版）が保存時と異なる: saved={saved.get('gpu_profile', 'official')} now={GPU_PROFILE}。"
                        "同じ種類の GPU（A100 なら A100）で再開する")
    return problems, warnings_


def build_resume_spec(manifest, local_ckpt_path):
    """保存時の run 設定をそのまま使い、resume 引数だけを付けた spec を作る（総 epoch 数・総 step 数を維持）"""
    saved = manifest["run_config"]
    spec = json.loads(json.dumps(saved))
    ov = spec["overrides"]
    ov["trainer.resume_mode"] = "resume_path"
    ov["trainer.resume_from_path"] = local_ckpt_path
    ov["model.partial_pretrain"] = get_base_model_local()          # 同じ revision を再取得（パスは環境で変わり得る）
    cur_sha, cur_path = current_data_sha256_for(saved["data_path"])
    assert cur_sha == saved["data_sha256"], "学習データが保存時と一致しない"
    train_path, train_sha, view = prepare_train_file(cur_path, cur_sha, manifest["world_size"])   # 保存時と同じ並べ替え・advantage を再作成
    sv = saved_training_view(saved, manifest["world_size"])
    assert view.get("enabled") == sv.get("enabled") and view.get("order_sha256") == sv.get("order_sha256") and view.get("advantage") == sv.get("advantage"), \
        "公式 8 GPU 再現の学習 file が保存時と一致しない"
    ov["data.train_files"] = train_path
    ov["data.val_files"] = ov["data.train_files"]
    ov["trainer.default_local_dir"] = f"{CKPT_DIR}/{spec['run_id']}"
    spec["base_local"] = ov["model.partial_pretrain"]
    spec["data_path"] = cur_path
    spec["train_file"], spec["train_file_sha256"] = train_path, train_sha
    spec["training_view"] = dict(view)
    spec.setdefault("logged_loss_scale", view.get("logged_loss_scale", 1.0))
    spec.setdefault("upload_steps", list(saved.get("expected_save_steps", [])))   # 旧版の spec は保存した全 step を転送していた
    spec.setdefault("upload_policy", "all")
    spec.setdefault("official_world_size", OFFICIAL_WORLD_SIZE)
    spec.setdefault("residual_differences", residual_differences(manifest["world_size"], saved.get("max_length", OFFICIAL_MAX_LENGTH), view.get("enabled", False)))
    spec["resume_from_path"] = local_ckpt_path
    spec["resumed_from"] = dict(step=manifest["step"], hf_repo=HF_CKPT_REPO_ID, folder_commit=manifest.get("folder_commit"))
    spec["fork_commit_at_resume"] = FORK_COMMIT
    spec["env"]["VERL_SFT_LOGGING_LEVEL"] = "INFO"
    assert ov["trainer.total_epochs"] == str(saved["epochs"]) and spec["total_steps"] == saved["total_steps"], "総 epoch 数・総 step 数は保存時の値を維持する"
    return spec


def resume_run_from_hf(run_id, step="latest", kill_after_step=None, upload=None):
    """HF の完了済み checkpoint から新プロセスで再開する。未保存 step は再実行になる"""
    require_training_env(None)
    rows = list_hf_checkpoints(run_id)
    complete = [r for r in rows if r["resumable"]]   # optimizer 状態込みで転送・検証済みのものだけが再開候補
    print(f"HF checkpoints for {run_id}:")
    for r in rows:
        print(f"  step {r['step']:6d} complete={r['complete']} content={r['content']} resumable={r['resumable']} files={r['n_files']} "
              f"commit={str(r.get('folder_commit', ''))[:10]} verified_at={r.get('verified_at', '')}")
    if not complete:
        raise SystemExit("再開できる checkpoint（optimizer 状態込みで転送・検証済み）が無い。不完全な転送や重みだけの step は候補にしない")
    if step == "latest":
        chosen = max(r["step"] for r in complete)
    else:
        chosen = int(step)
        assert chosen in [r["step"] for r in complete], f"step {chosen} は再開できる checkpoint ではない（未完了、または重みのみ）"
        newer = [r["step"] for r in complete if r["step"] > chosen]
        if newer:
            raise SystemExit(f"step {chosen} より新しい完了済み step {newer} がある。古い step から再開すると新しい step を上書きする恐れがあるので拒否する。"
                             "別 run として続ける場合は run_id を変えて手動で設定する")
    local, manifest, rev = download_hf_checkpoint(run_id, chosen)
    problems, warns = check_resume_compatibility(manifest)
    for w in warns:
        print("WARN:", w)
    if problems:
        for p in problems:
            print("PROBLEM:", p)
        raise SystemExit("再開条件を満たさない。条件を黙って変更しない")
    spec = build_resume_spec(manifest, local)
    total = spec["total_steps"]
    print(f"resume {run_id} from global_step_{chosen} (HF revision {rev[:10]}). 総 step {total} は維持。step {chosen + 1}..{total} を再実行（最後の保存以降の未保存 step は再実行になる）")
    print_run_spec(spec)
    return launch_training(spec, kill_after_step=kill_after_step, upload=upload)


# ---- FSDP checkpoint → HF 形式（公式 verl.model_merger をそのまま使う） ---------------------------
def merge_checkpoint(run_id, step, local_ckpt=None):
    """python -m verl.model_merger merge --backend fsdp（README の手順）で HF 形式へ変換する。戻り値: 変換先ディレクトリ"""
    ckpt = local_ckpt or f"{CKPT_DIR}/{run_id}/global_step_{step}"
    ok = False

    def _content(d):
        mp = os.path.join(d, "ao_ckpt_manifest.json")
        return json.load(open(mp)).get("content", "full") if os.path.isfile(mp) else "full"

    if os.path.isfile(os.path.join(ckpt, "fsdp_config.json")):
        world = json.load(open(os.path.join(ckpt, "fsdp_config.json")))["world_size"]
        ok, _ = checkpoint_is_complete(ckpt, world, require_stable=False, content=_content(ckpt))
    if not ok:   # ローカルに無い、または書きかけ → HF の完了済み checkpoint（重みのみでも変換はできる）を取得する
        ckpt, _, _ = download_hf_checkpoint(run_id, step)
        world = json.load(open(os.path.join(ckpt, "fsdp_config.json")))["world_size"]
        ok, _ = checkpoint_is_complete(ckpt, world, require_stable=False, content=_content(ckpt))
    assert ok, f"checkpoint が不完全: {ckpt}"
    target = f"{WORK_DIR}/merged/{run_id}/merged_step{step}"
    if os.path.isfile(os.path.join(target, "config.json")) and glob.glob(os.path.join(target, "*.safetensors")):
        print("merged model exists:", target)
        return target
    os.makedirs(target, exist_ok=True)
    cmd = [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir", ckpt, "--target_dir", target, "--trust_remote_code"]
    env = dict(os.environ, PYTHONPATH=REPO_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_DIR, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise RuntimeError("verl.model_merger failed")
    assert os.path.isfile(os.path.join(target, "config.json")) and glob.glob(os.path.join(target, "*.safetensors")), "変換結果に config/safetensors が無い"
    assert any(os.path.isfile(os.path.join(target, f)) for f in ("tokenizer.json", "tokenizer_config.json")), "変換結果に tokenizer が無い"
    print(f"merged {ckpt} -> {target} ({time.time() - t0:.0f}s)")
    return target


def get_manifest(run_id, step):
    """checkpoint の manifest（保存時の run 設定を含む）。ローカルに無ければ HF から manifest だけ取得する"""
    local = f"{CKPT_DIR}/{run_id}/global_step_{step}/ao_ckpt_manifest.json"
    if os.path.isfile(local):
        return json.load(open(local))
    p = hf_hub_download(HF_CKPT_REPO_ID, f"runs/{run_id}/global_step_{step}/ao_ckpt_manifest.json", repo_type="model", local_dir=f"{WORK_DIR}/hf_markers")
    return json.load(open(p))


# ---- 実験記録 ----------------------------------------------------------------------------------
def write_experiment_record(spec, results, evaluations=None, final_model=None, unverified=None, upload=True):
    rec = dict(
        run_id=spec["run_id"], kind=spec["kind"], written_at=now_iso(), session_started_at=SESSION_STARTED_AT,
        models=dict(teacher=TEACHER, base=dict(repo=spec["base_repo"], revision=spec["base_revision"]),
                    cot_student=dict(repo=spec["cot_repo"], revision=spec["cot_revision"], subfolder=spec["cot_subfolder"])),
        data=dict(repo=spec["dataset_repo"], revision=spec["dataset_revision"], raw_sha256=DATASET_EXPECTED_SHA256, ao_parquet_sha256=spec["data_sha256"],
                  rows=spec["n_rows"], target_style=spec["target_style"], ao_audit=AO_AUDIT_SUMMARY, prompt_note="prompt は公式と同一（step by step の指示を含む）。target は最後の \\boxed{} のみ"),
        length_and_mask=LENGTH_RECORD, training=dict(overrides=spec["overrides"], changes_vs_official=spec["changes_vs_official"], env=spec["env"],
                                                     total_steps=spec["total_steps"], steps_per_epoch=spec["steps_per_epoch"], warmup_steps=spec["warmup_steps"],
                                                     save_freq=spec["save_freq"], official_script=spec["official_script"], n_gpus=spec["n_gpus"],
                                                     official_world_size=spec.get("official_world_size"), training_view=spec.get("training_view"),
                                                     train_file=spec.get("train_file"), train_file_sha256=spec.get("train_file_sha256"),
                                                     logged_loss_scale=spec.get("logged_loss_scale", 1.0),
                                                     residual_differences_vs_official=spec.get("residual_differences")),
        code=dict(fork_repo=FORK_REPO_URL, fork_commit=FORK_COMMIT, upstream_reference_commit=UPSTREAM_REFERENCE_COMMIT, official_code_unchanged=OFFICIAL_CODE_UNCHANGED),
        environment=ENV_RECORD,
        runs=[dict(exit_code=r["exit_code"], killed_by_monitor=r["killed_by_monitor"], log_path=r["log_path"], wall_sec=r["wall_sec"], n_steps_logged=len(r["metrics"]),
                   first_step=(r["metrics"][0]["step"] if r["metrics"] else None), last_step=(r["metrics"][-1]["step"] if r["metrics"] else None),
                   last_loss_logged=(r["metrics"][-1].get("train/loss") if r["metrics"] else None),
                   last_loss_official_scale=(r["metrics"][-1].get("train/loss_official_scale") if r["metrics"] else None), timing=summarize_timing(r),
                   checkpoints_uploaded={s: dict(commit=v.get("commit"), marker_commit=v.get("marker_commit"), url=v.get("url")) for s, v in r["monitor"]["uploaded"].items()},
                   upload_failures=r["monitor"]["failed"], resumed_from=r["spec"].get("resumed_from")) for r in results],
        evaluations=evaluations or [], final_model=final_model,
        unverified=unverified or [],
        caveat="loss 低下だけで能力向上は主張しない。評価欄が空なら性能は未測定。",
    )
    path = f"{RECORD_DIR}/experiment_record_{spec['run_id']}.json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False, default=str)
    print("experiment record:", path)
    if upload and HF_API is not None and HF_CKPT_REPO_ID:
        ensure_ckpt_repo()
        ts = time.strftime("%Y%m%d-%H%M%S")
        for name in (f"experiment_record_{ts}.json", "experiment_record_latest.json"):
            HF_API.upload_file(path_or_fileobj=path, path_in_repo=f"runs/{spec['run_id']}/{name}", repo_id=HF_CKPT_REPO_ID, repo_type="model",
                               commit_message=f"{spec['run_id']} experiment record {ts}")
        print(f"uploaded to https://huggingface.co/{HF_CKPT_REPO_ID}/tree/main/runs/{spec['run_id']}")
    return rec


print("helpers ready: launch_training / TrainingMonitor / upload_checkpoint / list_hf_checkpoints / download_hf_checkpoint / resume_run_from_hf / write_experiment_record")
