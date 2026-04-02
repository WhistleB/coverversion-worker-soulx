"""
SoulX-Singer SVC Worker for RunPod Serverless.

Full pipeline:
  1. Download voice reference + song
  2. Preprocess (vocal separation + F0 extraction)
  3. SVC inference (zero-shot singing voice conversion)
  4. Post-processing (pedalboard reverb/EQ/compression)
  5. Upload result
"""

import os
import sys
import tempfile
import time
import subprocess
import traceback
import shutil

import requests
import runpod
import torchaudio
import numpy as np

SOULX_DIR = "/app/SoulX-Singer"
sys.path.insert(0, SOULX_DIR)


def download_file(url: str, dest_path: str):
    print(f"[Download] {url}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[Download] Done: {size_mb:.1f} MB")


def upload_file(file_path: str, filename: str, max_retries: int = 3) -> str:
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    print(f"[Upload] {filename} ({size_mb:.1f} MB)...")
    for attempt in range(1, max_retries + 1):
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(
                    "https://tmpfiles.org/api/v1/upload",
                    files={"file": (filename, f, "audio/wav")},
                    timeout=120,
                )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                raise RuntimeError(f"Response not success: {data}")
            url = data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
            print(f"[Upload] Done: {url}")
            return url
        except Exception as e:
            print(f"[Upload] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise RuntimeError(f"Upload failed after {max_retries} attempts: {e}")


def apply_post_fx(audio_path: str, vocal_volume: float = 1.3, reverb_amount: float = 0.25):
    """Apply pedalboard effects (reverb, compression, EQ)."""
    try:
        from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, Gain
        from pedalboard.io import AudioFile

        with AudioFile(audio_path) as f:
            sr = f.samplerate
            audio = f.read(f.frames)

        effects = [
            HighpassFilter(cutoff_frequency_hz=80),
            Compressor(threshold_db=-20, ratio=3.0, attack_ms=10, release_ms=100),
        ]
        if vocal_volume != 1.0:
            effects.append(Gain(gain_db=20 * np.log10(max(vocal_volume, 0.01))))
        if reverb_amount > 0:
            effects.append(Reverb(
                room_size=min(0.2 + reverb_amount * 0.8, 0.95),
                damping=min(0.5 + reverb_amount * 0.3, 0.85),
                wet_level=min(0.15 + reverb_amount * 0.55, 0.55),
                dry_level=max(0.7 - reverb_amount * 0.3, 0.45),
                width=0.8,
            ))

        processed = Pedalboard(effects)(audio, sr)
        peak = np.max(np.abs(processed))
        if peak > 1.0:
            processed = processed / peak * 0.95

        out_path = audio_path.replace(".wav", "_fx.wav")
        with AudioFile(out_path, "w", sr, processed.shape[0]) as f:
            f.write(processed)
        print(f"[FX] {len(effects)} effects applied")
        return out_path
    except Exception as e:
        print(f"[FX] Error: {e}")
        return audio_path


def run_soulx_svc(voice_path: str, song_path: str, output_dir: str,
                  pitch_shift: int = 0, n_steps: int = 32, cfg: float = 3.0):
    """
    Run SoulX-Singer SVC via webui_svc.py's pipeline (subprocess).
    Uses the built-in vocal separation and F0 extraction.
    """
    # Use the CLI inference which handles preprocessing internally
    # But CLI needs pre-extracted F0, so we use a custom script
    script = f"""
import sys, os, torch
sys.path.insert(0, '{SOULX_DIR}')
os.chdir('{SOULX_DIR}')

from pathlib import Path
from soulxsinger.preprocess.pipeline import PreprocessPipeline
from cli.inference_svc import build_model
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load('{SOULX_DIR}/soulxsinger/config/soulxsinger.yaml')

# Init preprocessor
pipeline = PreprocessPipeline(device='cuda')

# Preprocess prompt (voice reference)
prompt_save = Path('{output_dir}/preprocess/prompt')
prompt_save.mkdir(parents=True, exist_ok=True)
pipeline.configure(save_path=prompt_save)
p_ok, p_msg, prompt_wav, prompt_f0 = pipeline.run(
    audio_path=Path('{voice_path}'),
    vocal_sep=False,
    language='Mandarin',
)
print(f'Prompt preprocess: ok={{p_ok}}, wav={{prompt_wav}}, f0={{prompt_f0}}')
if not p_ok:
    print(f'Prompt failed: {{p_msg}}', file=sys.stderr)
    sys.exit(1)

# Preprocess target (song)
target_save = Path('{output_dir}/preprocess/target')
target_save.mkdir(parents=True, exist_ok=True)
pipeline.configure(save_path=target_save)
t_ok, t_msg, target_wav, target_f0 = pipeline.run(
    audio_path=Path('{song_path}'),
    vocal_sep=True,
    language='Mandarin',
)
print(f'Target preprocess: ok={{t_ok}}, wav={{target_wav}}, f0={{target_f0}}')
if not t_ok:
    print(f'Target failed: {{t_msg}}', file=sys.stderr)
    sys.exit(1)

# Build model
model = build_model(
    model_path='{SOULX_DIR}/pretrained_models/SoulX-Singer/model-svc.pt',
    config=config,
    device='cuda',
    use_fp16=True,
)

# Run SVC inference
import argparse
args = argparse.Namespace(
    device='cuda',
    model_path='{SOULX_DIR}/pretrained_models/SoulX-Singer/model-svc.pt',
    config='{SOULX_DIR}/soulxsinger/config/soulxsinger.yaml',
    prompt_wav_path=str(prompt_wav),
    target_wav_path=str(target_wav),
    prompt_f0_path=str(prompt_f0),
    target_f0_path=str(target_f0),
    save_dir='{output_dir}/generated',
    auto_shift=True,
    pitch_shift={pitch_shift},
    n_steps={n_steps},
    cfg={cfg},
    fp16=True,
    seed=42,
    auto_mix_acc=True,
)

from cli.inference_svc import process as svc_process
svc_process(args, config, model)

# Find generated file
gen_dir = Path('{output_dir}/generated')
wav_files = list(gen_dir.glob('*.wav'))
if wav_files:
    # Prefer mixed version if exists
    mixed = [f for f in wav_files if 'mixed' in f.name]
    result = mixed[0] if mixed else wav_files[0]
    # Copy to known location
    import shutil
    shutil.copy(result, '{output_dir}/final_output.wav')
    print(f'Output: {{result}}')
else:
    print('No output generated!', file=sys.stderr)
    sys.exit(1)
"""

    script_path = os.path.join(output_dir, "run_svc.py")
    with open(script_path, "w") as f:
        f.write(script)

    print(f"[SVC] Starting SoulX-Singer inference...")
    start = time.time()
    result = subprocess.run(
        ["python", script_path],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONPATH": f"{SOULX_DIR}:{os.environ.get('PYTHONPATH', '')}"},
    )
    elapsed = time.time() - start

    print(f"[SVC] Done in {elapsed:.1f}s, exit={result.returncode}")
    if result.stdout:
        print(f"[SVC] STDOUT:\n{result.stdout[-1000:]}")
    if result.stderr:
        print(f"[SVC] STDERR:\n{result.stderr[-500:]}")

    if result.returncode != 0:
        raise RuntimeError(f"SoulX-Singer failed: {result.stderr[-300:]}")

    output_path = os.path.join(output_dir, "final_output.wav")
    if not os.path.exists(output_path):
        raise RuntimeError("No output file generated")

    return output_path


def handler(job):
    job_input = job["input"]

    task_id = job_input.get("task_id", "unknown")
    song_url = job_input["song_url"]
    voice_url = job_input["voice_url"]
    pitch_shift = int(job_input.get("pitch_shift", 0))
    n_steps = int(job_input.get("n_steps", 32))
    cfg = float(job_input.get("cfg", 3.0))
    vocal_volume = float(job_input.get("vocal_volume", 1.3))
    reverb = float(job_input.get("reverb", 0.25))

    print(f"\n{'='*60}")
    print(f"[Job] task_id={task_id}, pitch={pitch_shift}, steps={n_steps}, cfg={cfg}")
    print(f"[Job] vocal_vol={vocal_volume}, reverb={reverb}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            total_start = time.time()

            # Stage 1: Download
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "downloading", "progress": 0.05
            })
            t = time.time()
            song_path = os.path.join(tmpdir, "song.wav")
            voice_path = os.path.join(tmpdir, "voice.wav")
            download_file(song_url, song_path)
            download_file(voice_url, voice_path)
            download_time = time.time() - t

            song_info = torchaudio.info(song_path)
            song_duration = song_info.num_frames / song_info.sample_rate
            print(f"[Job] Song: {song_duration:.1f}s, Download: {download_time:.1f}s")

            # Stage 2: SVC (includes separation + F0 + inference + mixing)
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "converting", "progress": 0.15
            })
            t = time.time()
            svc_output_dir = os.path.join(tmpdir, "svc_output")
            os.makedirs(svc_output_dir, exist_ok=True)
            output_path = run_soulx_svc(
                voice_path, song_path, svc_output_dir,
                pitch_shift=pitch_shift, n_steps=n_steps, cfg=cfg,
            )
            svc_time = time.time() - t
            print(f"[Job] SVC: {svc_time:.1f}s")

            # Stage 3: Post FX
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "mixing", "progress": 0.85
            })
            if vocal_volume != 1.0 or reverb > 0:
                output_path = apply_post_fx(output_path, vocal_volume, reverb)

            # Stage 4: Upload
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "uploading", "progress": 0.95
            })
            output_info = torchaudio.info(output_path)
            output_duration = output_info.num_frames / output_info.sample_rate
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

            t = time.time()
            output_url = upload_file(output_path, f"cover_{task_id}.wav")
            upload_time = time.time() - t

            total_time = time.time() - total_start
            print(f"\n[Job] === SUMMARY ===")
            print(f"[Job] Download:  {download_time:.1f}s")
            print(f"[Job] SVC:       {svc_time:.1f}s")
            print(f"[Job] Upload:    {upload_time:.1f}s")
            print(f"[Job] TOTAL:     {total_time:.1f}s")
            print(f"[Job] Output:    {output_duration:.1f}s, {output_size_mb:.1f} MB")

            return {
                "task_id": task_id,
                "status": "success",
                "output_url": output_url,
                "duration": round(output_duration, 2),
                "svc_time": round(svc_time, 2),
                "total_time": round(total_time, 2),
                "output_format": "wav",
                "sample_rate": output_info.sample_rate,
                "size_mb": round(output_size_mb, 2),
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
            }


if __name__ == "__main__":
    print("[Init] SoulX-Singer SVC Worker")
    runpod.serverless.start({"handler": handler})
