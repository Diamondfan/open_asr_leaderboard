#!/usr/bin/env python3
"""Transcribe audio files listed in a JSONL file using a registered provider."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers import get_provider


def main():
    parser = argparse.ArgumentParser(description="Transcribe audios from a JSONL file")
    parser.add_argument("--jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--model_name", default="microsoft/azure-speech-05-2026", help="Model name (provider/variant)")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: input with .results.jsonl suffix)")
    parser.add_argument("--prompt", default=None, help="Optional prompt for the provider")
    args = parser.parse_args()

    output_path = args.output or args.jsonl.replace(".jsonl", ".results.jsonl")

    provider, variant = get_provider(args.model_name)

    results = []
    with open(args.jsonl, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    for entry in entries:
        wav_path = entry["WavPath"]
        uuid = entry["UUID"]
        locale = entry.get("locale", "en-US")
        language = locale.split("-")[0]  # "en-US" -> "en"

        print(f"Transcribing: {uuid} ({wav_path})")

        kwargs = dict(use_url=False, language=language)
        if args.prompt:
            kwargs["prompt"] = args.prompt

        try:
            text = provider.transcribe(variant, wav_path, {}, **kwargs)
        except Exception as e:
            print(f"  ERROR: {e}")
            text = ""

        entry["Prediction"] = text
        print(f"  Result: {text}")
        results.append(entry)

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
