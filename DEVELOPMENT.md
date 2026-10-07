# Working on the SDK

The public entrypoint is `KokoroMultispeaker`. One instance owns the original stock weights and selects a language adapter for each request. Keep that instance loaded when narrating several passages.

## Reading the code

| Module | Responsibility |
| --- | --- |
| `shared.py` | Public Python API, model loading and narration |
| `bundles.py` | Pinned downloads, language catalog, checksum and bundle validation |
| `cli.py` | Command-line arguments; loads only the requested language |
| `language_weights.py` | Serialized adapter selection and shared stock tensor ownership |
| `frontend.py` | English/German text and phoneme conventions |
| `word_plan.py` | Source-word ownership without changing phonemes |
| `context.py` | Production one-word context and trimming |
| `native_runtime.py` | Native duration, F0, excitation and decoder execution |
| `native_models.py`, `native_decoder.py`, `native_utils.py` | Checkpoint-compatible native model layers |
| `releases.json` | Immutable download revisions, checksums and voice inventories |

The native layers match the model checkpoints. Preserve their tensor operations, state keys and random-noise order when changing the surrounding API.

## Local development

Use Python 3.10 and the CPU PyTorch 2.6.0 wheel:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
ruff check kokoro_multispeaker tests
black --check kokoro_multispeaker tests --extend-exclude '/context.py$'
python -m pytest -q
```

The context helper is shared with AkinVox production; keep its behavior consistent. Unit tests cover adapter isolation, stock restoration, shared weights, chunk ownership, cancellation and language download selection. They do not download model weights. Model checks also cover installed Python/CLI output, pinned downloads and the published samples.

Add later languages to the same project. Each needs its frontend, verified stock-compatible adapter, finalized voicepacks and immutable registry entry, with the same context and stock-isolation checks. Existing pins and voices remain independently identifiable.
