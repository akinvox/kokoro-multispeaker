"""Generate fixed-speaker speech using a native language adapter."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="AkinVox Kokoro multilingual native adapters"
    )
    parser.add_argument(
        "--text", help="Text to speak; long passages are chunked automatically"
    )
    parser.add_argument("--text-file", help="UTF-8 input file containing a passage")
    parser.add_argument("--language", choices=["de", "en", "stock"], default="de")
    parser.add_argument("--speaker")
    parser.add_argument("--output", default="speech.wav")
    parser.add_argument(
        "--adapter-dir",
        help="Local language folder with adapter.pt, config.json and SHA256SUMS",
    )
    parser.add_argument(
        "--base-dir", help="Local pinned stock base, config and voices folder"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--context-before",
        default="",
        help="One preceding word from an adjacent narration request",
    )
    parser.add_argument(
        "--context-after",
        default="",
        help="One following word from an adjacent narration request",
    )
    parser.add_argument("--list-speakers", action="store_true")
    args = parser.parse_args()
    if args.list_speakers:
        import json
        from pathlib import Path

        if args.language == "stock":
            names = ["af_heart"]
        elif args.adapter_dir:
            from .bundles import _checked_bundle

            release, _ = _checked_bundle(args.adapter_dir, args.language)
            names = release["speaker_names"]
        else:
            registry = json.loads(Path(__file__).with_name("releases.json").read_text())
            if args.language not in registry:
                parser.error("No published " + args.language + " adapter")
            names = registry[args.language]["speaker_names"]
        print("\n".join(names))
        return
    if bool(args.text) == bool(args.text_file):
        parser.error("Supply exactly one of --text or --text-file")
    from pathlib import Path
    from .frontend import spoken_text

    try:
        if (
            args.text_file
            and Path(args.text_file).resolve() == Path(args.output).resolve()
        ):
            raise ValueError("Input text and output audio must use different paths")
        text = (
            Path(args.text_file).read_text(encoding="utf-8")
            if args.text_file
            else args.text
        )
        text = spoken_text(text, "narration")
        if len(text) > 10000:
            raise ValueError("Text exceeds 10,000 characters; split it into passages")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, str(error) + "\n")

    from .shared import KokoroMultispeaker

    try:
        language = "de" if args.language == "stock" else args.language
        adapters = {language: args.adapter_dir} if args.adapter_dir else None
        model = KokoroMultispeaker(
            adapters,
            args.base_dir,
            args.device,
            args.threads,
            languages=[language] if adapters is None else None,
        )
        model.save(
            text,
            args.output,
            args.language,
            args.speaker,
            args.seed,
            context_before=args.context_before,
            context_after=args.context_after,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, str(error) + "\n")
    print(f"Saved {args.output} (24000 Hz)")


if __name__ == "__main__":
    main()
