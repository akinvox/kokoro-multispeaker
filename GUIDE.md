# Using the voices

Choose a voice, select English or German, and provide text. Both languages share the original Kokoro base. Their adapter folders contain the learned language weights and ready-to-use voicepacks.

The package downloads the selected adapters from a fixed Hugging Face revision and checks their hashes.

## Install on Linux or WSL

Tested with Python 3.10 and PyTorch 2.6.0. Install matching eSpeak-NG **1.51** library and data for German.

```bash
sudo apt-get install espeak-ng libespeak-ng1 libsndfile1 python3-venv
espeak-ng --version  # Check that this reports 1.51.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install https://github.com/akinvox/kokoro-multispeaker/releases/download/2026-10-07.1/akinvox_kokoro_multispeaker-1.0.1-py3-none-any.whl
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

The first model load downloads the pinned original Kokoro base. The matching adapters are downloaded automatically. Keep the instance loaded for repeated use. If you need only one language, use `KokoroMultispeaker(languages=["en"], device="cpu")` or `languages=["de"]` to avoid downloading and loading the other adapter. The CLI already loads only the requested language.

If your distribution provides a different eSpeak version, install the library and matching data from [eSpeak-NG 1.51, following its upstream build instructions](https://github.com/espeak-ng/espeak-ng/blob/1.51/docs/building.md). A separate installation can be selected explicitly:

```bash
export KOKORO_ESPEAK_LIBRARY="/path/to/espeak-1.51/lib/libespeak-ng.so.1"
export KOKORO_ESPEAK_DATA="/path/to/espeak-1.51/share/espeak-ng-data"
```

Use the actual paths from your installation. German inference checks the loaded library version and rejects a mismatch.

For NVIDIA GPU inference, install the PyTorch CUDA build instead of the CPU build:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Then use `device="cuda"` in Python or `--device cuda` on the command line.

## Python

```python
from kokoro_multispeaker import KokoroMultispeaker

model = KokoroMultispeaker(device="cpu")

print(model.speakers["en"])
print(model.speakers["de"])

model.save("Welcome to our next story.", "english.wav",
           language="en", speaker="AV_ariala")
model.save("Willkommen zu unserer nächsten Geschichte.", "german.wav",
           language="de", speaker="AV_thorsten")
```

`synthesize()` returns a NumPy audio array. `save()` writes a mono 24-kHz WAV. `stream()` yields chunks as they finish, with timings relative to each chunk where word ownership is available.

## Command line

```bash
kokoro-multispeaker --language en --list-speakers
kokoro-multispeaker --language de --list-speakers

kokoro-multispeaker --language en \
  --speaker AV_helen --text "Welcome to our next story." --output english.wav

kokoro-multispeaker --language de \
  --speaker AV_kerstin --text "Willkommen zu unserer nächsten Geschichte." --output german.wav
```

Use `--text-file passage.txt` to read a UTF-8 passage instead of `--text`. Longer passages are split automatically, with one neighboring word at each chunk boundary.

## Voice IDs

| English | German |
| --- | --- |
| `AV_ariala` | `AV_thorsten` |
| `AV_brody` | `AV_bernd` |
| `AV_david` | `AV_eva_k` |
| `AV_elliott` | `AV_karlsson` |
| `AV_ralph` | `AV_kerstin` |
| `AV_cori` | `AV_mucky` |
| `AV_phil` | `AV_hokuspokus` |
| `AV_helen` | `AV_julia` |

Each standalone `voices/AV_<name>.pt` has shape `[510, 1, 256]`. Its decoder and predictor conditioning is finalized in every row. These packs require their matching language adapter; they are not interchangeable with unadapted stock voicepacks.

## Context between adjacent requests

For separate requests that belong to one narration, optionally provide one preceding and one following word:

```python
audio = model.synthesize(
    "She opened the letter and began to read.",
    language="en",
    speaker="AV_cori",
    context_before="Then",
    context_after="Outside",
)
```

The neighboring words are context only and are removed from the returned audio. Automatic internal chunking already uses one word on each side; these options are for adjacent requests. If exact source-word ownership cannot be established, inference returns the body alone and does not speak the optional neighbors. The chunk exposes `context_used=False` and no word timings in that case.

Use `stream()` to inspect each chunk's `context_used` and `timings`; `synthesize()` returns only the concatenated audio array.

The equivalent CLI options are `--context-before` and `--context-after`.

## Offline use

Before going offline, install dependencies and the English spaCy model. Download `adapters/en` and `adapters/de` from the [dated model snapshot](https://huggingface.co/AKinvox/kokoro-multispeaker/tree/2026-10-07). In Python supply `adapter_dirs={"en": "adapters/en", "de": "adapters/de"}`; the CLI accepts `--adapter-dir adapters/en` or `--adapter-dir adapters/de`. Save these files from the official Kokoro base revision `f3ff3571791e39611d31c381e3a41a3af07b4987`:

```text
stock/
  config.json
  kokoro-v1_0.pth
  voices/
    af_heart.pt
```

Pass `base_dir="stock"` in Python or `--base-dir stock` on the command line. The loader verifies the base and adapter checksums.

Original stock American-English `af_heart` is available through `language="stock"`, with the language adapter turned off. The convenience interface exposes this stock voice only.

Requests are limited to 10,000 characters and 180 seconds of generated audio. Split books into passages.
