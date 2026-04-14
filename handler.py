"""
SoulX-Singer SVC Worker for RunPod Serverless.

Full pipeline:
  1. Download voice reference + song
  2. Demucs vocal separation (external, same as Seed-VC)
  3. SoulX preprocess (F0 extraction)
  4. SVC inference (zero-shot singing voice conversion)
  5. Post-processing (pedalboard reverb/EQ on vocals ONLY)
  6. Mix processed vocals + clean instrumental (ffmpeg amix normalize=0)
  7. MP3 conversion + metadata
  8. Upload result
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


def separate_vocals(song_path: str, output_dir: str):
    """Separate vocals and instrumental using demucs (same as Seed-VC)."""
    print(f"[Demucs] Separating vocals...")
    cmd = [
        "python", "-m", "demucs",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-o", output_dir,
        song_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[-300:]}")

    song_name = os.path.splitext(os.path.basename(song_path))[0]
    separated_dir = os.path.join(output_dir, "htdemucs", song_name)
    vocals_path = os.path.join(separated_dir, "vocals.wav")
    instrumental_path = os.path.join(separated_dir, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        raise RuntimeError(f"Vocals not found: {os.listdir(separated_dir)}")

    print(f"[Demucs] Done.")
    return vocals_path, instrumental_path


def mix_audio(vocals_path: str, instrumental_path: str, output_path: str,
              vocal_volume: float = 1.0, reverb: float = 0.0):
    """
    Apply FX to vocals ONLY, then mix with clean instrumental.
    Same approach as Seed-VC worker.
    """
    from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, Gain
    from pedalboard.io import AudioFile

    print(f"[Mix] Processing: vocal_vol={vocal_volume}, reverb={reverb}")

    # ── Step 1: Process vocals ──
    with AudioFile(vocals_path) as f:
        vocal_sr = f.samplerate
        vocal_audio = f.read(f.frames)

    # Fade-in/out to prevent start/end pops
    fade_samples = int(vocal_sr * 0.5)  # 500ms
    if vocal_audio.shape[-1] > fade_samples * 2:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        for ch in range(vocal_audio.shape[0]):
            vocal_audio[ch, :fade_samples] *= fade_in
            vocal_audio[ch, -fade_samples:] *= fade_out

    vocal_effects = [
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(threshold_db=-20, ratio=3.0, attack_ms=10, release_ms=100),
    ]
    if vocal_volume != 1.0:
        vocal_effects.append(Gain(gain_db=20 * np.log10(max(vocal_volume, 0.01))))
    if reverb > 0:
        vocal_effects.append(Reverb(
            room_size=min(0.2 + reverb * 0.8, 0.95),
            damping=min(0.5 + reverb * 0.3, 0.85),
            wet_level=min(0.15 + reverb * 0.55, 0.55),
            dry_level=max(0.7 - reverb * 0.3, 0.45),
            width=0.8,
        ))

    processed_vocal = Pedalboard(vocal_effects)(vocal_audio, vocal_sr)
    processed_vocal_path = vocals_path.replace('.wav', '_fx.wav')
    with AudioFile(processed_vocal_path, 'w', vocal_sr, processed_vocal.shape[0]) as f:
        f.write(processed_vocal)
    print(f"[Mix] Vocal effects applied: {len(vocal_effects)} effects")

    # ── Step 2: Mix with ffmpeg (normalize=0 to keep loudness) ──
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_vocal_path,
        "-i", instrumental_path,
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=longest:weights=1 1:normalize=0",
        "-ac", "2", "-ar", "44100",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mix failed: {result.stderr[-300:]}")
    print(f"[Mix] Done.")


def run_soulx_svc(voice_path: str, vocals_path: str, output_dir: str,
                  pitch_shift: int = 0, n_steps: int = 32, cfg: float = 3.0,
                  seed: int = 42, auto_shift: bool = True):
    """
    Run SoulX-Singer SVC on pre-separated vocals (not raw song).
    Voice reference preprocessing still uses SoulX's built-in pipeline.
    """
    script = f"""
import sys, os, torch
sys.path.insert(0, '{SOULX_DIR}')
os.chdir('{SOULX_DIR}')

from pathlib import Path
from preprocess.pipeline import PreprocessPipeline
from cli.inference_svc import build_model
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load('{SOULX_DIR}/soulxsinger/config/soulxsinger.yaml')

# Init preprocessor
pipeline = PreprocessPipeline(
    device='cuda',
    language='Mandarin',
    save_dir='{output_dir}/preprocess',
    vocal_sep=False,
    max_merge_duration=60000,
    midi_transcribe=False,
)

# Preprocess prompt (voice reference) — no separation needed, already clean
prompt_save = Path('{output_dir}/preprocess/prompt')
prompt_save.mkdir(parents=True, exist_ok=True)
pipeline.save_dir = str(prompt_save)
pipeline.vocal_sep = False  # voice reference is already clean
pipeline.run(
    audio_path=str(Path('{voice_path}')),
    vocal_sep=False,
    max_merge_duration=60000,
    language='Mandarin',
)
prompt_wav = prompt_save / 'vocal.wav'
prompt_f0 = prompt_save / 'vocal_f0.npy'
if not prompt_wav.exists() or not prompt_f0.exists():
    print(f'Prompt preprocess failed: missing {{prompt_wav}} or {{prompt_f0}}', file=sys.stderr)
    sys.exit(1)
print(f'Prompt preprocess done')

import gc
gc.collect()
torch.cuda.empty_cache()

# Preprocess target (demucs-separated vocals) — no separation needed
target_save = Path('{output_dir}/preprocess/target')
target_save.mkdir(parents=True, exist_ok=True)
pipeline.save_dir = str(target_save)
pipeline.vocal_sep = False  # already separated by demucs
pipeline.run(
    audio_path=str(Path('{vocals_path}')),
    vocal_sep=False,
    max_merge_duration=60000,
    language='Mandarin',
)
target_wav = target_save / 'vocal.wav'
target_f0 = target_save / 'vocal_f0.npy'
if not target_wav.exists() or not target_f0.exists():
    print(f'Target preprocess failed: missing {{target_wav}} or {{target_f0}}', file=sys.stderr)
    sys.exit(1)
print(f'Target preprocess done')

gc.collect()
torch.cuda.empty_cache()

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
    auto_shift={auto_shift},
    pitch_shift={pitch_shift},
    n_steps={n_steps},
    cfg={cfg},
    use_fp16=True,
    seed={seed},
    auto_mix_acc=False,  # We handle mixing externally with ffmpeg
)

from cli.inference_svc import process as svc_process
svc_process(args, config, model)

# Find generated vocal file
gen_dir = Path('{output_dir}/generated')
generated = gen_dir / 'generated.wav'
if not generated.exists():
    wav_files = list(gen_dir.glob('*.wav'))
    if wav_files:
        generated = wav_files[0]
    else:
        print('No output generated!', file=sys.stderr)
        sys.exit(1)

# Copy to known location
import shutil
shutil.copy(str(generated), '{output_dir}/svc_vocals.wav')
print(f'SVC vocals output ready')
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

    output_path = os.path.join(output_dir, "svc_vocals.wav")
    if not os.path.exists(output_path):
        raise RuntimeError("No SVC output file generated")

    return output_path


def handler(job):
    job_input = job["input"]

    # Warmup
    if job_input.get("mode") == "warmup":
        print("[Warmup] Worker is warm and ready.")
        return {"status": "warm", "message": "Worker is ready"}

    task_id = job_input.get("task_id", "unknown")
    song_url = job_input["song_url"]
    voice_url = job_input["voice_url"]
    pitch_shift = int(job_input.get("pitch_shift", 0))
    n_steps = int(job_input.get("n_steps", 32))
    cfg = float(job_input.get("cfg", 3.0))
    seed = int(job_input.get("seed", 42))
    auto_shift = bool(job_input.get("auto_shift", True))
    vocal_volume = float(job_input.get("vocal_volume", 1.3))
    reverb = float(job_input.get("reverb", 0.25))
    output_format = job_input.get("output_format", "mp3_192")
    cover_image = job_input.get("cover_image", "")
    artist_name = job_input.get("artist_name", "")
    song_title = job_input.get("song_title", "")

    print(f"\n{'='*60}")
    print(f"[Job] task_id={task_id}, pitch={pitch_shift}, steps={n_steps}, cfg={cfg}, seed={seed}")
    print(f"[Job] auto_shift={auto_shift}, vocal_vol={vocal_volume}, reverb={reverb}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            total_start = time.time()

            # ── Stage 1: Download ────────────────────────────────
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

            # ── Stage 2: Demucs vocal separation ─────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "separating", "progress": 0.1
            })
            t = time.time()
            vocals_path, instrumental_path = separate_vocals(song_path, os.path.join(tmpdir, "demucs_out"))
            separation_time = time.time() - t
            print(f"[Job] Separation: {separation_time:.1f}s")

            # ── Stage 3: SoulX SVC (preprocess + inference) ──────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "converting", "progress": 0.2
            })
            t = time.time()
            svc_output_dir = os.path.join(tmpdir, "svc_output")
            os.makedirs(svc_output_dir, exist_ok=True)
            svc_vocals_path = run_soulx_svc(
                voice_path, vocals_path, svc_output_dir,
                pitch_shift=pitch_shift, n_steps=n_steps, cfg=cfg,
                seed=seed, auto_shift=auto_shift,
            )
            svc_time = time.time() - t
            print(f"[Job] SVC: {svc_time:.1f}s")

            # ── Stage 4: FX on vocals + Mix with instrumental ───
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "mixing", "progress": 0.85
            })
            t = time.time()
            final_output = os.path.join(tmpdir, "final_cover.wav")
            mix_audio(svc_vocals_path, instrumental_path, final_output,
                      vocal_volume=vocal_volume, reverb=reverb)
            mix_time = time.time() - t

            # ── Stage 5: Format conversion ───────────────────────
            t = time.time()
            output_info = torchaudio.info(final_output)
            output_duration = output_info.num_frames / output_info.sample_rate

            if output_format in ("mp3_320", "mp3_192"):
                bitrate = "320k" if output_format == "mp3_320" else "192k"
                mp3_output = final_output.replace(".wav", ".mp3")

                cover_path = None
                if cover_image:
                    cover_url = f"https://raw.githubusercontent.com/WhistleB/coverversion-worker/main/assets/covers/{cover_image}.png"
                    cover_path = os.path.join(tmpdir, "cover.png")
                    try:
                        download_file(cover_url, cover_path)
                    except Exception:
                        cover_path = None

                metadata_args = []
                if artist_name:
                    metadata_args += ["-metadata", f"artist={artist_name}"]
                if song_title:
                    metadata_args += ["-metadata", f"title={song_title}"]
                metadata_args += ["-metadata", "album=AI Cover"]

                if cover_path and os.path.exists(cover_path):
                    convert_cmd = [
                        "ffmpeg", "-y", "-i", final_output, "-i", cover_path,
                        "-map", "0:a", "-map", "1",
                        "-c:a", "libmp3lame", "-b:a", bitrate,
                        "-c:v", "png", "-disposition:v", "attached_pic",
                        "-id3v2_version", "3",
                    ] + metadata_args + [mp3_output]
                else:
                    convert_cmd = ["ffmpeg", "-y", "-i", final_output, "-b:a", bitrate] + metadata_args + [mp3_output]

                subprocess.run(convert_cmd, capture_output=True, timeout=60)
                if os.path.exists(mp3_output):
                    final_output = mp3_output

            output_size_mb = os.path.getsize(final_output) / (1024 * 1024)
            file_ext = os.path.splitext(final_output)[1]
            format_time = time.time() - t

            # ── Stage 6: Upload ──────────────────────────────────
            runpod.serverless.progress_update(job, {
                "task_id": task_id, "stage": "uploading", "progress": 0.95
            })
            t = time.time()
            output_url = upload_file(final_output, f"cover_{task_id}{file_ext}")
            upload_time = time.time() - t

            total_time = time.time() - total_start
            print(f"\n[Job] === SUMMARY ===")
            print(f"[Job] Download:   {download_time:.1f}s")
            print(f"[Job] Separation: {separation_time:.1f}s")
            print(f"[Job] SVC:        {svc_time:.1f}s")
            print(f"[Job] Mix:        {mix_time:.1f}s")
            print(f"[Job] Format:     {format_time:.1f}s")
            print(f"[Job] Upload:     {upload_time:.1f}s")
            print(f"[Job] TOTAL:      {total_time:.1f}s")
            print(f"[Job] Output:     {output_duration:.1f}s, {output_size_mb:.1f} MB")

            return {
                "task_id": task_id,
                "status": "success",
                "output_url": output_url,
                "duration": round(output_duration, 2),
                "separation_time": round(separation_time, 2),
                "svc_time": round(svc_time, 2),
                "mix_time": round(mix_time, 2),
                "format_time": round(format_time, 2),
                "upload_time": round(upload_time, 2),
                "total_time": round(total_time, 2),
                "output_format": output_format,
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
    print("[Init] SoulX-Singer SVC Worker v2")
    runpod.serverless.start({"handler": handler})
