"""Fake stand-in for verl.trainer.fsdp_sft_trainer_ours used ONLY to test the notebook's launch/monitor/log-parsing logic on CPU.
Mimics: console metric lines, checkpoint directory layout + save order, tracker file, resume_path handling, and SIGTERM.
Args are hydra-style key=value (only a few are read)."""
import os, sys, time, json, re, torch


def main():
    ov = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out = ov["trainer.default_local_dir"]
    total_epochs = int(ov["trainer.total_epochs"])
    save_freq = int(ov["trainer.save_freq"])
    steps_per_epoch = 2
    total = steps_per_epoch * total_epochs
    resume_mode = ov.get("trainer.resume_mode", "disable")
    step = 0
    if resume_mode == "resume_path":
        p = ov["trainer.resume_from_path"]
        assert os.path.isdir(p), p
        step = int(re.search(r"global_step_(\d+)", p).group(1))
        for f in ("model_world_size_1_rank_0.pt", "optim_world_size_1_rank_0.pt", "extra_state_world_size_1_rank_0.pt", "data.pt"):
            assert os.path.isfile(os.path.join(p, f)), f
        print(f"[Rank 0] Loaded model from {p}/model_world_size_1_rank_0.pt", flush=True)
        print(f"[Rank 0] Loaded optimizer from {p}/optim_world_size_1_rank_0.pt", flush=True)
        print(f"[Rank 0] Loaded rng from {p}/extra_state_world_size_1_rank_0.pt", flush=True)
        print(f"[Rank 0] Loaded lr_scheduler from {p}/extra_state_world_size_1_rank_0.pt", flush=True)
        print(f"[Rank 0] Successfully loaded dataloader state from {p}/data.pt", flush=True)
    print(f"Normalize batch size by dp 1", flush=True)
    print(f"Number of steps/epoch {steps_per_epoch}, number of epochs {total_epochs}, total number of steps {total}", flush=True)
    print(f"[Rank 0] Total training steps: {total},", flush=True)
    time.sleep(1.0)  # "init"
    import math
    warm = int(total * 0.1)

    def lr_at(s):
        if s < warm:
            return 5e-5 * s / max(1, warm)
        prog = (s - warm) / max(1, total - warm)
        return 5e-5 * max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))

    while step < total:
        step += 1
        time.sleep(0.6)
        print(f"ratio_list_all type: <class 'list'>", flush=True)
        print(f"step:{step} - train/loss:{1.0 / step:.6f} - train/lr:{lr_at(step):.8e} - train/grad_norm:0.5 - train/clip_frac_low:0.0 - train/clip_frac_high:0.0 - train/entropy/mean:0.1 - train/ratio/mean:0.0 - train/ratio_clip/mean:0.0", flush=True)
        if step % save_freq == 0 or step == total:
            d = os.path.join(out, f"global_step_{step}")
            print(f"Saving checkpoint to: {d}", flush=True)
            os.makedirs(os.path.join(d, "huggingface"), exist_ok=True)
            for f in ("model_world_size_1_rank_0.pt", "optim_world_size_1_rank_0.pt", "extra_state_world_size_1_rank_0.pt"):
                torch.save({"step": step, "blob": torch.zeros(2000)}, os.path.join(d, f))
                time.sleep(0.3)
            json.dump({"FSDP_version": 2, "world_size": 1}, open(os.path.join(d, "fsdp_config.json"), "w"))
            json.dump({"model_type": "fake"}, open(os.path.join(d, "huggingface", "config.json"), "w"))
            json.dump({"tokenizer_class": "fake"}, open(os.path.join(d, "huggingface", "tokenizer_config.json"), "w"))
            open(os.path.join(d, "huggingface", "tokenizer.json"), "w").write("{}")
            torch.save({"pos": step}, os.path.join(d, "data.pt"))
            with open(os.path.join(out, "latest_checkpointed_iteration.txt"), "w") as f:
                f.write(str(step))
            print(f"Updated checkpoint tracker: {out}/latest_checkpointed_iteration.txt", flush=True)
    print("Final validation metrics: None", flush=True)


if __name__ == "__main__":
    main()
