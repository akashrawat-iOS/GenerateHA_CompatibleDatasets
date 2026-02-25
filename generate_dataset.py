"""
generate_dataset.py
-------------------
Generate paired training data for speech enhancement fine-tuning.

Pipeline per sample:
  1. Load a clean 16 kHz mono WAV file.
  2. Randomly select a babble noise segment and mix at a random SNR (0–20 dB).
  3. Resample the noisy mixture from 16 kHz → 8 kHz.
  4. Apply a 3.4 kHz low-pass filter (telephone bandwidth).
  5. Encode/decode through G.729D via FFmpeg subprocess.
  6. Save:
       - degraded 8 kHz signal  → data/degraded/<id>.wav
       - original clean 16 kHz  → data/clean/<id>.wav
  7. Log (id, snr_db, noise_file) to data/log.csv.

Usage:
  python generate_dataset.py \
      --clean_dir  path/to/clean_speech \
      --noise_dir  path/to/musan_babble \
      --output_dir data \
      --snr_min 0 --snr_max 20 \
      --workers 4 \
      --seed 42
"""

import argparse
import csv
import logging
import os
import random
import subprocess
import tempfile
from multiprocessing import Pool, Manager
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def _load_mono_16k(path: str) -> np.ndarray:
    """Load a WAV file as a 16 kHz mono float32 array.

    Raises ValueError if the file is not 16 kHz or not mono.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != 16_000:
        raise ValueError(f"Expected 16 kHz, got {sr} Hz: {path}")
    if audio.ndim > 1:
        # Downmix to mono
        audio = audio.mean(axis=1)
    return audio


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------

def add_babble(clean: np.ndarray, noise_files: list[str], rng: random.Random) -> tuple[np.ndarray, str]:
    """Randomly select a babble noise file and crop/tile it to match *clean* length.

    Parameters
    ----------
    clean:
        Clean speech array (16 kHz, mono, float32).
    noise_files:
        List of paths to babble noise WAV files.
    rng:
        A seeded :class:`random.Random` instance for reproducibility.

    Returns
    -------
    noise : np.ndarray
        Noise segment aligned to the length of *clean*.
    noise_path : str
        Path of the selected noise file.
    """
    noise_path = rng.choice(noise_files)
    noise, sr = sf.read(noise_path, dtype="float32", always_2d=False)
    if sr != 16_000:
        raise ValueError(f"Noise file must be 16 kHz, got {sr} Hz: {noise_path}")
    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    target_len = len(clean)
    if len(noise) < target_len:
        # Tile noise to cover the full clean length
        repeats = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeats)
    # Random crop
    max_start = len(noise) - target_len
    start = rng.randint(0, max_start)
    noise = noise[start : start + target_len]
    return noise, noise_path


def mix_with_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix *clean* and *noise* at the requested SNR, then normalise to avoid clipping.

    Parameters
    ----------
    clean:
        Clean speech array.
    noise:
        Noise array (same length as *clean*).
    snr_db:
        Target signal-to-noise ratio in dB.

    Returns
    -------
    mixed : np.ndarray
        The normalised mixture.
    """
    eps = 1e-10
    clean_rms = np.sqrt(np.mean(clean ** 2) + eps)
    noise_rms = np.sqrt(np.mean(noise ** 2) + eps)

    # Scale noise to achieve target SNR
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixed = clean + noise_scaled

    # Normalise to peak amplitude ≤ 0.99 to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)

    return mixed.astype(np.float32)


def resample_to_8k(audio_16k: np.ndarray) -> np.ndarray:
    """Downsample a 16 kHz signal to 8 kHz using scipy (polyphase filter).

    Parameters
    ----------
    audio_16k:
        Input signal sampled at 16 000 Hz.

    Returns
    -------
    audio_8k : np.ndarray
        Output signal sampled at 8 000 Hz (float32).
    """
    from scipy.signal import resample_poly

    audio_8k = resample_poly(audio_16k, up=1, down=2).astype(np.float32)
    return audio_8k


def apply_lowpass(audio: np.ndarray, cutoff_hz: float = 3400.0, fs: float = 8000.0, order: int = 8) -> np.ndarray:
    """Apply a Butterworth low-pass filter to simulate telephone bandwidth (3.4 kHz).

    Parameters
    ----------
    audio:
        Input audio array.
    cutoff_hz:
        Low-pass cutoff frequency in Hz. Defaults to 3400 Hz.
    fs:
        Sampling rate of *audio*. Defaults to 8000 Hz.
    order:
        Filter order. Defaults to 8.

    Returns
    -------
    filtered : np.ndarray
        Filtered audio array (float32).
    """
    nyq = fs / 2.0
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    filtered = sosfilt(sos, audio).astype(np.float32)
    return filtered


def g729_process(audio_8k: np.ndarray, fs: int = 8000) -> np.ndarray:
    """Encode and decode *audio_8k* through the G.729 codec via FFmpeg.

    FFmpeg uses the ``libcodec2`` or ``g729`` codec (libbcg729 / bcg729) when
    available.  The encode→decode round-trip introduces the characteristic
    G.729 artefacts used to simulate telephone speech.

    If FFmpeg with G.729 support is unavailable, the function falls back to
    returning the input unchanged and emits a warning.

    Parameters
    ----------
    audio_8k:
        Input signal at 8 kHz, float32 in [-1, 1].
    fs:
        Sampling rate (must be 8000 Hz for G.729).

    Returns
    -------
    decoded : np.ndarray
        Audio after G.729 encode/decode round-trip (float32, 8 kHz).
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_wav = os.path.join(tmp, "input.wav")
        out_wav = os.path.join(tmp, "output.wav")

        sf.write(in_wav, audio_8k, fs, subtype="PCM_16")

        # Build FFmpeg command for G.729 encode → decode round-trip.
        # Requires FFmpeg compiled with libbcg729 (G.729 codec support).
        cmd = [
            "ffmpeg", "-y",
            "-i", in_wav,
            "-ar", str(fs),
            "-ac", "1",
            "-c:a", "g729",          # encode with G.729
            "-b:a", "8k",
            os.path.join(tmp, "encoded.g729"),
        ]
        decode_cmd = [
            "ffmpeg", "-y",
            "-f", "g729",
            "-ar", str(fs),
            "-ac", "1",
            "-i", os.path.join(tmp, "encoded.g729"),
            "-c:a", "pcm_s16le",
            out_wav,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            subprocess.run(decode_cmd, check=True, capture_output=True)
            decoded, _ = sf.read(out_wav, dtype="float32", always_2d=False)
            # Align length with input (codec may add/remove a few samples)
            decoded = _align_length(decoded, len(audio_8k))
            return decoded.astype(np.float32)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning(
                "G.729 processing failed (%s). "
                "Returning low-pass filtered signal without codec artefacts. "
                "Install FFmpeg with libbcg729 support to enable G.729.",
                exc,
            )
            return audio_8k


def _align_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Crop or zero-pad *audio* to exactly *target_len* samples."""
    if len(audio) >= target_len:
        return audio[:target_len]
    pad = np.zeros(target_len - len(audio), dtype=audio.dtype)
    return np.concatenate([audio, pad])


def save_pair(
    file_id: str,
    clean_16k: np.ndarray,
    degraded_8k: np.ndarray,
    output_dir: str,
) -> None:
    """Write the clean 16 kHz and degraded 8 kHz WAV files to *output_dir*.

    Directory structure::

        <output_dir>/
            clean/<file_id>.wav      (16 kHz, PCM_16)
            degraded/<file_id>.wav   (8 kHz,  PCM_16)

    Parameters
    ----------
    file_id:
        Stem used as the filename for both outputs (without extension).
    clean_16k:
        Clean speech array (16 kHz).
    degraded_8k:
        Degraded/processed array (8 kHz).
    output_dir:
        Root output directory.
    """
    clean_dir = os.path.join(output_dir, "clean")
    degraded_dir = os.path.join(output_dir, "degraded")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(degraded_dir, exist_ok=True)

    sf.write(os.path.join(clean_dir, f"{file_id}.wav"), clean_16k, 16_000, subtype="PCM_16")
    sf.write(os.path.join(degraded_dir, f"{file_id}.wav"), degraded_8k, 8_000, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Per-sample worker
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> dict | None:
    """Process a single clean speech file.

    This function is designed to be called from a :class:`multiprocessing.Pool`.

    Parameters
    ----------
    args : tuple
        ``(clean_path, noise_files, output_dir, snr_min, snr_max, seed_offset)``

    Returns
    -------
    dict or None
        Log entry ``{id, snr_db, noise_file}`` on success, ``None`` on failure.
    """
    clean_path, noise_files, output_dir, snr_min, snr_max, seed_offset = args

    file_id = Path(clean_path).stem
    # Per-sample deterministic RNG derived from the global seed
    rng = random.Random(seed_offset)

    try:
        # 1. Load clean speech
        clean_16k = _load_mono_16k(clean_path)

        # 2. Add babble noise at random SNR
        noise, noise_path = add_babble(clean_16k, noise_files, rng)
        snr_db = rng.uniform(snr_min, snr_max)
        noisy_16k = mix_with_snr(clean_16k, noise, snr_db)

        # 3. Resample 16 k → 8 k
        noisy_8k = resample_to_8k(noisy_16k)

        # 4. Low-pass filter at 3.4 kHz
        noisy_8k_lp = apply_lowpass(noisy_8k)

        # 5. G.729 encode/decode
        degraded_8k = g729_process(noisy_8k_lp)

        # 6. Save pair
        save_pair(file_id, clean_16k, degraded_8k, output_dir)

        logger.info("Processed %s | SNR=%.1f dB | noise=%s", file_id, snr_db, os.path.basename(noise_path))
        return {"id": file_id, "snr_db": round(snr_db, 3), "noise_file": noise_path}

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to process %s: %s", clean_path, exc)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paired clean/degraded training data for speech enhancement."
    )
    parser.add_argument("--clean_dir", required=True, help="Directory of clean 16 kHz WAV files.")
    parser.add_argument("--noise_dir", required=True, help="Directory of babble noise WAV files (e.g. MUSAN).")
    parser.add_argument("--output_dir", default="data", help="Root output directory (default: data).")
    parser.add_argument("--snr_min", type=float, default=0.0, help="Minimum SNR in dB (default: 0).")
    parser.add_argument("--snr_max", type=float, default=20.0, help="Maximum SNR in dB (default: 20).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42).")
    args = parser.parse_args()

    # Set global random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Gather input files
    clean_files = sorted(Path(args.clean_dir).glob("**/*.wav"))
    noise_files = sorted(Path(args.noise_dir).glob("**/*.wav"))

    if not clean_files:
        logger.error("No WAV files found in --clean_dir: %s", args.clean_dir)
        raise SystemExit(1)
    if not noise_files:
        logger.error("No WAV files found in --noise_dir: %s", args.noise_dir)
        raise SystemExit(1)

    noise_files_str = [str(p) for p in noise_files]
    logger.info("Found %d clean files and %d noise files.", len(clean_files), len(noise_files))

    # Build per-sample argument tuples with unique seed offsets
    worker_args = [
        (
            str(p),
            noise_files_str,
            args.output_dir,
            args.snr_min,
            args.snr_max,
            args.seed + idx,           # unique offset per sample
        )
        for idx, p in enumerate(clean_files)
    ]

    # Process in parallel
    log_rows: list[dict] = []
    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_process_one, worker_args):
            if result is not None:
                log_rows.append(result)

    # Write CSV log
    log_path = os.path.join(args.output_dir, "log.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "snr_db", "noise_file"])
        writer.writeheader()
        writer.writerows(sorted(log_rows, key=lambda r: r["id"]))

    logger.info(
        "Done. Processed %d/%d files. Log saved to %s.",
        len(log_rows),
        len(clean_files),
        log_path,
    )


if __name__ == "__main__":
    main()
