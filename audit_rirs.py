"""
audit_rirs.py
-------------
One-time RIR audit script.

Computes RT60 proxy using energy decay (Schroeder-style integration)
and filters RIR files into a validated pool.

Usage:
    python audit_rirs.py \
        --rir_dir path/to/musan_rir \
        --rt60_min 0.2 \
        --rt60_max 0.6 \
        --output_file valid_rirs.txt
"""

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def estimate_rt60_proxy(rir: np.ndarray, sr: int) -> float:
    """
    Estimate RT60 using energy decay (Schroeder integration proxy).
    Uses -30 dB crossing and doubles it (EDT approximation).
    """
    energy = rir ** 2

    # Schroeder reverse integration
    decay = np.cumsum(energy[::-1])[::-1]

    # Normalize
    decay /= (decay[0] + 1e-10)

    decay_db = 10 * np.log10(decay + 1e-10)

    # Find -30 dB crossing
    idx = np.where(decay_db <= -30)[0]
    if len(idx) == 0:
        return len(rir) / sr  # fallback to duration

    t_30 = idx[0] / sr
    return t_30 * 2  # extrapolate to RT60


def main():
    parser = argparse.ArgumentParser(description="Audit and filter RIR files based on RT60 proxy.")
    parser.add_argument("--rir_dir", required=True, help="Directory containing RIR WAV files.")
    parser.add_argument("--rt60_min", type=float, default=0.2, help="Minimum acceptable RT60 (seconds).")
    parser.add_argument("--rt60_max", type=float, default=0.6, help="Maximum acceptable RT60 (seconds).")
    parser.add_argument("--output_file", default="valid_rirs.txt", help="Output file for valid RIR paths.")
    args = parser.parse_args()

    rir_files = sorted(Path(args.rir_dir).glob("**/*.wav"))

    if not rir_files:
        print("No RIR files found.")
        return

    valid_paths = []
    rt60_values = []

    print(f"Found {len(rir_files)} RIR files. Auditing...")

    for path in rir_files:
        try:
            rir, sr = sf.read(path, dtype="float32", always_2d=False)

            if rir.ndim > 1:
                rir = rir.mean(axis=1)

            if sr != 16000:
                rir = resample_poly(rir, up=16000, down=sr).astype(np.float32)
                sr = 16000

            rt60 = estimate_rt60_proxy(rir, sr)
            rt60_values.append(rt60)

            if args.rt60_min <= rt60 <= args.rt60_max:
                valid_paths.append(str(path))

        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Save valid RIR paths
    with open(args.output_file, "w") as f:
        for p in valid_paths:
            f.write(p + "\n")

    print("\n--- Audit Summary ---")
    print(f"Total RIRs: {len(rir_files)}")
    print(f"Valid RIRs: {len(valid_paths)}")
    print(f"RT60 range observed: {min(rt60_values):.3f}s – {max(rt60_values):.3f}s")
    print(f"Saved valid RIR list to: {args.output_file}")


if __name__ == "__main__":
    main()
