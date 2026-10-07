# AkinVox Kokoro Multispeaker

**8 English voices. 8 German voices. Language adapters for the original Kokoro model.**

Choose a voice and give it text. Both languages share one Kokoro base, and the matching adapter is selected automatically. `AV_` means an AkinVox voice.

I'm building this for [AkinVox](https://akinvox.com), my audiobook site. I wanted more voices and languages for narration while keeping Kokoro small and easy to run. English and German are the first two languages. More adapters and voices will follow here.

[Listen to every voice on Hugging Face](https://huggingface.co/AKinvox/kokoro-multispeaker#listen).

## Get started

Follow the [installation guide](GUIDE.md#install-on-linux-or-wsl) for CPU or NVIDIA GPU setup. It covers Python dependencies and the eSpeak-NG 1.51 requirement for German.

```python
from kokoro_multispeaker import KokoroMultispeaker

model = KokoroMultispeaker(device="cpu")
model.save(
    "A new story begins at the little bookshop beside the station.",
    "english.wav",
    language="en",
    speaker="AV_cori",
)
model.save(
    "Eine neue Geschichte beginnt im kleinen Buchladen am Bahnhof.",
    "german.wav",
    language="de",
    speaker="AV_julia",
)
```

The first load downloads the base and adapters. Keep the model loaded to read more passages. To load one language, use `KokoroMultispeaker(languages=["en"])` or `languages=["de"]`.

You can also use the command line:

```bash
kokoro-multispeaker --language en --list-speakers
kokoro-multispeaker --language en --speaker AV_cori \
  --text "Welcome to our next story." --output speech.wav
```

## Voices

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

The [model card](https://huggingface.co/AKinvox/kokoro-multispeaker) has a player, passage and training-data hours for each voice. Training data was capped at 20 hours per training voice; held-out recordings are excluded from those figures.

## Longer passages

Long text is split automatically. Each chunk uses one neighboring word on either side, then trims those words from the output. This is the same approach I use in AkinVox production.

```python
for chunk in model.stream(long_text, language="en", speaker="AV_brody"):
    audio = chunk.audio  # Mono float32 audio at 24 kHz.
```

The [guide](GUIDE.md) covers streaming, context between requests and offline use. Voicepacks are ready to use with their matching adapter. Pronunciation and pacing can vary, so listen to several passages before choosing a voice for a longer narration.

AkinVox is still taking shape. Feedback on the voices, pronunciation and storytelling rhythm is welcome.

## Development and licence

See [DEVELOPMENT.md](DEVELOPMENT.md) for the module map and local checks. AkinVox code and adapter weights use Apache-2.0. [Third-party notices](THIRD_PARTY.md) list upstream licences and recording sources.
