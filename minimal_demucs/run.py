import argparse
from pathlib import Path

import torch
import torchaudio

from demucs import (
    load_model,
    preprocess_and_normalize_audio,
    run_model_with_splits_and_shifts,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks", nargs="+", type=Path, default=[])
    parser.add_argument(
        "-d",
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use, default is cuda if available else cpu",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Load in model from checkpoint
    model = load_model("demucs_quantized").to(args.device)

    # Initialize output path
    out = Path("separated") / "demucs_quantized"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Separated tracks will be stored in {out.resolve()}")

    for track in args.tracks:
        if not track.exists():
            print(f"File {track} does not exist.")
            continue

        print(f"Separating track {track}")

        # Load audio and run_model
        wav, sr = torchaudio.load(str(track))
        wav = wav.to(args.device)
        wav, ref = preprocess_and_normalize_audio(
            wav, sr, model.audio_channels, model.samplerate
        )
        sources = run_model_with_splits_and_shifts(model, wav, split=True)
        sources = sources * ref.std() + ref.mean()

        # Save outputs
        track_folder = out / track.name.rsplit(".", 1)[0]
        track_folder.mkdir(exist_ok=True)
        for source, name in zip(sources, model.sources):
            source = source / max(1.01 * source.abs().max(), 1)
            source = source.cpu()
            wavname = str(track_folder / f"{name}.wav")
            torchaudio.save(wavname, source, sample_rate=model.samplerate)


if __name__ == "__main__":
    main()
