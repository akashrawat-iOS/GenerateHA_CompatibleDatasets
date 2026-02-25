# GenerateHA_CompatibleDatasets

Generate paired **clean / degraded** training data for speech-enhancement
fine-tuning (e.g. for hearing-aid or telephony models).

## Pipeline

For every clean speech file the script:

1. Loads a 16 kHz mono WAV.
2. Picks a random babble-noise segment (e.g. from the MUSAN *babble* subset).
3. Mixes clean + noise at a randomly chosen SNR in **[0 dB, 20 dB]**.
4. Resamples the mixture **16 kHz → 8 kHz**.
5. Applies a **3.4 kHz low-pass filter** (telephone bandwidth).
6. Encodes and decodes the 8 kHz signal through the **G.729D codec** via
   FFmpeg (requires FFmpeg compiled with `libbcg729`).
7. Saves the paired outputs and logs SNR / noise-file to a CSV.

```
data/
├── clean/          # original 16 kHz speech  (training targets)
├── degraded/       # G.729-processed 8 kHz mixture  (model inputs)
└── log.csv         # id, snr_db, noise_file
```

## Requirements

```
pip install soundfile scipy numpy
```

G.729 encoding requires **FFmpeg ≥ 4.x** compiled with `libbcg729`.  
If G.729 support is unavailable the script falls back to saving the
low-pass filtered signal and logs a warning.

## Usage

```bash
python generate_dataset.py \
    --clean_dir  /path/to/clean_speech \
    --noise_dir  /path/to/musan/noise/free-sound \
    --output_dir data \
    --snr_min 0 \
    --snr_max 20 \
    --workers 4 \
    --seed 42
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--clean_dir` | *(required)* | Directory of clean 16 kHz mono WAV files |
| `--noise_dir` | *(required)* | Directory of babble noise WAV files |
| `--output_dir` | `data` | Root output directory |
| `--snr_min` | `0.0` | Minimum SNR in dB |
| `--snr_max` | `20.0` | Maximum SNR in dB |
| `--workers` | `4` | Parallel worker processes |
| `--seed` | `42` | Global random seed for reproducibility |

## Module API

| Function | Description |
|---|---|
| `add_babble(clean, noise_files, rng)` | Select & align a babble noise segment |
| `mix_with_snr(clean, noise, snr_db)` | Mix at target SNR, normalise to avoid clipping |
| `resample_to_8k(audio_16k)` | Polyphase resample 16 kHz → 8 kHz |
| `apply_lowpass(audio, cutoff_hz, fs)` | Butterworth low-pass filter |
| `g729_process(audio_8k)` | G.729 encode/decode round-trip via FFmpeg subprocess |
| `save_pair(file_id, clean_16k, degraded_8k, output_dir)` | Write paired WAV files |