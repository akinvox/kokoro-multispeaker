# Third-party notices

| Component | Role | License / source |
| --- | --- | --- |
| Kokoro / Kokoro-82M | Model lineage and native architecture | [Apache-2.0](https://github.com/hexgrad/kokoro/blob/main/LICENSE) |
| StyleTTS2 | Native architecture code ancestry | MIT; original notice in `licenses/StyleTTS2-MIT.txt` |
| eSpeak-NG 1.51 | German phonemization, installed separately | [GPL-3.0-or-later](https://github.com/espeak-ng/espeak-ng/blob/1.51/COPYING) |
| phonemizer-fork 3.3.2 | Python interface to eSpeak, installed separately | GPL-3.0-or-later; see the installed package license |
| PyTorch | Inference | BSD-style; see the installed package license |
| Transformers / Hugging Face Hub / safetensors | Model architecture and download/loading | Apache-2.0; see their package licenses |
| NumPy / SciPy / SoundFile | Numerical and WAV support | See their installed package licenses |

The Apache-2.0 release license does not replace these dependency licenses. No ASR model is required for this fixed-speaker release.

## German recording sources

Voice IDs refer to learned synthetic voicepacks. Source recordings are not redistributed, and no speaker or dataset creator endorsement is implied. Audio cleaning and adaptation are modifications made by AkinVox.

| Released voice IDs | Source / attribution | Source terms |
| --- | --- | --- |
| `AV_thorsten` | [Thorsten-Voice](https://www.thorsten-voice.de/ueber-das-projekt/aufnahmen/), Thorsten Müller and Dominik Kreutz | CC0; retained neutral recordings |
| `AV_bernd`, `AV_eva_k`, `AV_karlsson`, `AV_hokuspokus`, `AV_julia` | [HUI-Audio-Corpus-German](https://opendata.iisys.de/dataset/hui-audio-corpus-german/), Pascal Puchtler, Johannes Wirth and René Peinl, IISYS / Hof University; LibriVox-derived recordings | CC0 as listed in the authors' [paper, Table 1](https://arxiv.org/pdf/2106.06309); preserve speaker and corpus attribution |
| `AV_kerstin` | [Rhasspy Kerstin](https://github.com/rhasspy/dataset-voice-kerstin), volunteer speaker and Michael Hansen / Rhasspy | CC0-1.0 |
| `AV_mucky` | [Mucky Voice German Speech Dataset](https://huggingface.co/datasets/Muckylixx/mucky-voice-final), Maurice Hartmann / Muckylixx; retained revision `0766729d5497e4a79b9c2e8bec03da6ebc0454e2` | CC BY 4.0; attribution and modification disclosure retained |

Both language adapters use original released stock Kokoro v1.0. No source audio or training population is redistributed.

## Release licenses

AkinVox’s inference code and exported weights are released under Apache-2.0. Upstream code, dependencies and source datasets retain their own terms and notices. No training audio is redistributed.


## English recording sources and frontend

English recording sources include LibriVox readers David Wales (6454), Elliott Miller (3717), Gregg Margarite (3490), Ralph Snelson (2140), Cori Samuel (92), Phil Benson (6097), John Van Stan (9017), and Helen Taylor (9136). Reader credits and source IDs are retained for attribution; no endorsement is implied. See [LibriVox](https://librivox.org/) for the source project. Training recordings are not distributed.

Misaki 0.9.4 (Apache-2.0), spaCy and its English model (MIT), and espeakng-loader are separately installed frontend dependencies. Refer to each installed distribution for its complete notices.
