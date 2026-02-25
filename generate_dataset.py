"""
generate_dataset.py

Generates paired training data for speech enhancement fine-tuning.

Pipeline per sample:
  1. Load clean 16 kHz mono WAV.
  2. Randomly select a babble noise segment of the same length.
  3. Mix clean + noise at a randomly chosen SNR in [SNR_MIN_DB, SNR_MAX_DB].
  4. Resample the noisy mixture from 16 kHz to 8 kHz.
  5. Apply a low-pass filter at 3.4 kHz to simulate telephone bandwidth.
  6. Encode/decode through G.729D via an external binary or FFmpeg.
  7. Save:
       data/degraded/<id>.wav  – 8 kHz degraded signal
       data/clean/<id>.wav     – 16 kHz clean reference (target)
  8. Append a row to data/log.csv (filename, snr_db, noise_file).

Usage:
    python generate_dataset.py \
        --clean_dir  path/to/clean_speech \
        --noise_dir  path/to/musan_babble \
        --output_dir data \
        --workers    4 \
        --seed       42
"""

import argparse
import csv
import logging
import multiprocessing
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLEAN_SR = 16_000       # Hz – sample rate of clean input files
DEGRADED_SR = 8_000     # Hz – target sample rate after downsampling
LOWPASS_CUTOFF = 3_400  # Hz – telephone-band low-pass cutoff
SNR_MIN_DB = 0          # dB  – minimum SNR for mixing
SNR_MAX_DB = 20         # dB  – maximum SNR for mixing
LOWPASS_ORDER = 6       # Butterworth filter order

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def add_babble(
    clean: np.ndarray,
    noise_files: List[Path],
    rng: random.Random,
) -> Tuple[np.ndarray, Path]:
    """
    Randomly select a babble noise file and crop/loop it to match the length
    of *clean*.

    Parameters
    ----------
    clean:       1-D float32 array of the clean speech signal.
    noise_files: List of candidate noise file paths.
    rng:         Seeded :class:`random.Random` instance for reproducibility.

    Returns
    -------
    noise_segment : 1-D float32 ndarray with the same length as *clean*.
    chosen_file   : Path of the selected noise file.
    """
    chosen = Path(rng.choice(noise_files))
    noise_data, noise_sr = sf.read(str(chosen), dtype="float32", always_2d=False)

    # Mix down to mono if stereo
    if noise_data.ndim == 2:
        noise_data = noise_data.mean(axis=1)

    # Resample noise to CLEAN_SR if necessary
    if noise_sr != CLEAN_SR:
        noise_data = _resample(noise_data, noise_sr, CLEAN_SR)

    target_len = len(clean)
    noise_len = len(noise_data)

    if noise_len >= target_len:
        # Randomly crop
        start = rng.randint(0, noise_len - target_len)
        noise_segment = noise_data[start : start + target_len]
    else:
        # Loop / tile until long enough, then crop
        repeats = int(np.ceil(target_len / noise_len))
        noise_segment = np.tile(noise_data, repeats)[:target_len]

    return noise_segment.astype(np.float32), chosen


def mix_with_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """
    Mix *clean* and *noise* at the desired SNR level and normalise the result
    to avoid clipping.

    Parameters
    ----------
    clean:  1-D float32 array of the clean speech.
    noise:  1-D float32 array of babble noise (same length as *clean*).
    snr_db: Target signal-to-noise ratio in dB.

    Returns
    -------
    mixed : 1-D float32 array, peak-normalised to [-1, 1).
    """
    # Power of the clean signal
    clean_power = np.mean(clean ** 2)

    if clean_power < 1e-10:
        # Near-silent utterance – return clean as-is
        return clean.copy()

    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        # Near-silent noise – return clean
        return clean.copy()

    # Scale noise to achieve the requested SNR:  SNR = 10*log10(Ps / Pn_scaled)
    # => Pn_scaled = Ps / 10^(SNR/10)
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise * noise_scale

    mixed = clean + scaled_noise

    # Prevent clipping by peak-normalising
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    return mixed.astype(np.float32)


def resample_to_8k(signal: np.ndarray, orig_sr: int = CLEAN_SR) -> np.ndarray:
    """
    Resample *signal* from *orig_sr* to DEGRADED_SR (8 kHz).

    Uses :func:`scipy.signal.resample_poly` for high-quality polyphase
    resampling without requiring integer ratio upsampling.

    Parameters
    ----------
    signal:  1-D float32 array.
    orig_sr: Original sample rate of the input signal.

    Returns
    -------
    resampled : 1-D float32 array at DEGRADED_SR.
    """
    return _resample(signal, orig_sr, DEGRADED_SR)


def apply_lowpass(signal: np.ndarray, sr: int = DEGRADED_SR) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter at LOWPASS_CUTOFF Hz to *signal* to
    simulate the telephone bandwidth (G.711 / G.729 typical bandwidth).

    Parameters
    ----------
    signal: 1-D float32 array.
    sr:     Sample rate of *signal* in Hz.

    Returns
    -------
    filtered : 1-D float32 array.
    """
    nyquist = sr / 2.0
    cutoff_norm = LOWPASS_CUTOFF / nyquist  # normalised to [0, 1]
    sos = butter(LOWPASS_ORDER, cutoff_norm, btype="low", output="sos")
    filtered = sosfilt(sos, signal)
    return filtered.astype(np.float32)


def g729_process(signal: np.ndarray, sr: int = DEGRADED_SR) -> np.ndarray:
    """
    Encode and decode *signal* using the G.729D codec via an external binary
    (``g729codec``) or FFmpeg with ``libg729`` / ``bcg729``.

    The function tries the following in order:
      1. A binary named ``g729codec`` on PATH (expected interface:
         ``g729codec encode <in.wav> <out.g729>`` /
         ``g729codec decode <in.g729> <out.wav>``).
      2. FFmpeg with the ``g729`` codec (available when compiled with
         ``--enable-libg729`` / ``bcg729``).
      3. If neither is available, log a warning and return *signal* unchanged
         so the rest of the pipeline can still produce output.

    Parameters
    ----------
    signal: 1-D float32 array at *sr* Hz.
    sr:     Sample rate (should be DEGRADED_SR = 8 000 Hz for G.729).

    Returns
    -------
    processed : 1-D float32 array after encode → decode round-trip.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_wav = os.path.join(tmp, "input.wav")
        out_wav = os.path.join(tmp, "output.wav")
        g729_file = os.path.join(tmp, "encoded.g729")

        # Write the 8 kHz mono WAV to a temp file (PCM s16le)
        _write_wav(in_wav, signal, sr)

        # --- Attempt 1: dedicated g729codec binary ---
        if shutil.which("g729codec"):
            try:
                subprocess.run(
                    ["g729codec", "encode", in_wav, g729_file],
                    check=True, capture_output=True,
                )
                subprocess.run(
                    ["g729codec", "decode", g729_file, out_wav],
                    check=True, capture_output=True,
                )
                result, _ = sf.read(out_wav, dtype="float32", always_2d=False)
                return _match_length(result, len(signal))
            except subprocess.CalledProcessError as exc:
                logger.warning("g729codec failed: %s", exc.stderr)

        # --- Attempt 2: FFmpeg with g729 codec ---
        if shutil.which("ffmpeg"):
            try:
                # Encode: WAV → G.729 raw bitstream
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", in_wav,
                        "-ar", str(sr),
                        "-ac", "1",
                        "-c:a", "g729",
                        "-b:a", "8k",
                        g729_file,
                    ],
                    check=True, capture_output=True,
                )
                # Decode: G.729 raw bitstream → WAV
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ar", str(sr),
                        "-ac", "1",
                        "-c:a", "g729",
                        "-i", g729_file,
                        out_wav,
                    ],
                    check=True, capture_output=True,
                )
                result, _ = sf.read(out_wav, dtype="float32", always_2d=False)
                return _match_length(result, len(signal))
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "FFmpeg G.729 encode/decode failed: %s", exc.stderr
                )

        logger.warning(
            "No G.729 codec found (tried 'g729codec' and 'ffmpeg'). "
            "Returning unprocessed signal."
        )
        return signal.copy()


def save_pair(
    file_id: str,
    clean_16k: np.ndarray,
    degraded_8k: np.ndarray,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Save a clean/degraded pair to disk under *output_dir*.

    Saves:
      <output_dir>/clean/<file_id>.wav     at CLEAN_SR
      <output_dir>/degraded/<file_id>.wav  at DEGRADED_SR

    Parameters
    ----------
    file_id:     Base name for both output files (no extension).
    clean_16k:   1-D float32 array at CLEAN_SR.
    degraded_8k: 1-D float32 array at DEGRADED_SR.
    output_dir:  Root output directory (``data/`` by default).

    Returns
    -------
    clean_path, degraded_path : Paths of the saved files.
    """
    clean_dir = output_dir / "clean"
    degraded_dir = output_dir / "degraded"
    clean_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    clean_path = clean_dir / f"{file_id}.wav"
    degraded_path = degraded_dir / f"{file_id}.wav"

    _write_wav(str(clean_path), clean_16k, CLEAN_SR)
    _write_wav(str(degraded_path), degraded_8k, DEGRADED_SR)

    return clean_path, degraded_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling using scipy."""
    from math import gcd
    from scipy.signal import resample_poly

    if orig_sr == target_sr:
        return signal.copy()

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled = resample_poly(signal, up, down)
    return resampled.astype(np.float32)


def _write_wav(path: str, signal: np.ndarray, sr: int) -> None:
    """Write a float32 array as a 16-bit PCM WAV file."""
    # Clip to valid range before converting
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    sf.write(path, pcm, sr, subtype="PCM_16")


def _match_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Truncate or zero-pad *arr* to *target_len* samples."""
    arr = arr.flatten()
    if len(arr) >= target_len:
        return arr[:target_len].astype(np.float32)
    pad = np.zeros(target_len - len(arr), dtype=np.float32)
    return np.concatenate([arr, pad])


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _process_file(args: Tuple) -> Optional[dict]:
    """
    Worker function executed in a subprocess.

    Returns a dict with logging info on success, or None on failure.
    """
    (
        clean_path,
        noise_files,
        output_dir,
        seed,
    ) = args

    file_id = Path(clean_path).stem

    # Per-file RNG seeded deterministically for reproducibility
    rng = random.Random(seed)
    snr_db = rng.uniform(SNR_MIN_DB, SNR_MAX_DB)

    try:
        # 1. Load clean speech
        clean, sr = sf.read(str(clean_path), dtype="float32", always_2d=False)
        if clean.ndim == 2:
            clean = clean.mean(axis=1)
        if sr != CLEAN_SR:
            clean = _resample(clean, sr, CLEAN_SR)

        # 2. Add babble noise segment
        noise_segment, noise_file = add_babble(clean, noise_files, rng)

        # 3. Mix at chosen SNR
        mixed = mix_with_snr(clean, noise_segment, snr_db)

        # 4. Resample to 8 kHz
        mixed_8k = resample_to_8k(mixed, CLEAN_SR)

        # 5. Apply low-pass filter
        filtered_8k = apply_lowpass(mixed_8k, DEGRADED_SR)

        # 6. G.729 encode/decode round-trip
        degraded_8k = g729_process(filtered_8k, DEGRADED_SR)

        # 7. Save pair
        save_pair(file_id, clean, degraded_8k, output_dir)

        logger.info(
            "Processed %s | SNR=%.1f dB | noise=%s",
            file_id, snr_db, noise_file.name,
        )

        return {
            "filename": file_id,
            "snr_db": round(snr_db, 2),
            "noise_file": str(noise_file),
        }

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to process %s: %s", file_id, exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _collect_wav_files(directory: Path) -> List[Path]:
    return sorted(directory.rglob("*.wav"))


def _write_log_header(log_path: Path) -> None:
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "snr_db", "noise_file"])
        writer.writeheader()


def _append_log_row(log_path: Path, row: dict) -> None:
    with open(log_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "snr_db", "noise_file"])
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paired speech-enhancement training data."
    )
    parser.add_argument(
        "--clean_dir",
        required=True,
        type=Path,
        help="Directory containing clean 16 kHz mono WAV files.",
    )
    parser.add_argument(
        "--noise_dir",
        required=True,
        type=Path,
        help="Directory containing babble noise WAV files (e.g., MUSAN babble).",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("data"),
        type=Path,
        help="Root output directory. Defaults to 'data/'.",
    )
    parser.add_argument(
        "--workers",
        default=max(1, multiprocessing.cpu_count() - 1),
        type=int,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Global random seed for reproducibility.",
    )
    args = parser.parse_args()

    # Seed the global RNG (multiprocessing workers use per-file seeds derived
    # from this seed + file index so results are fully reproducible regardless
    # of worker ordering).
    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_files = _collect_wav_files(args.clean_dir)
    noise_files = _collect_wav_files(args.noise_dir)

    if not clean_files:
        raise SystemExit(f"No WAV files found in --clean_dir: {args.clean_dir}")
    if not noise_files:
        raise SystemExit(f"No WAV files found in --noise_dir: {args.noise_dir}")

    logger.info(
        "Found %d clean file(s) and %d noise file(s).",
        len(clean_files),
        len(noise_files),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "log.csv"
    _write_log_header(log_path)

    # Build per-file worker arguments; derive a per-file seed for reproducibility.
    worker_args = [
        (
            clean_files[i],
            noise_files,
            args.output_dir,
            args.seed + i,          # unique per-file seed
        )
        for i in range(len(clean_files))
    ]

    # Process in parallel
    with multiprocessing.Pool(processes=args.workers) as pool:
        results = pool.map(_process_file, worker_args)

    # Write CSV log (in original file order)
    succeeded = 0
    for row in results:
        if row is not None:
            _append_log_row(log_path, row)
            succeeded += 1

    logger.info(
        "Done. %d/%d files processed successfully. Log: %s",
        succeeded,
        len(clean_files),
        log_path,
    )


if __name__ == "__main__":
    main()
