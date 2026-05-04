"""
Syntho-Limb NS2C Engine: Sprint 0 - Video Pre-processing
=========================================================
Strips metadata from raw camera footage and re-encodes to a clean MP4
that downstream tools (TRIBE v2, MediaPipe) can read without issues.
"""

import argparse

import cv2
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def strip_metadata(input_path: str, output_path: str) -> None:
    print(f"[*] Reading raw video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("[!] Error: could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] {n_frames} frames  {width}x{height}  {fps:.1f} fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("[*] Scrubbing metadata and rewriting frames ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[*] Saved clean video to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip video metadata.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", help="Override raw video path")
    parser.add_argument("--output", help="Override clean video path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    import os

    raw = args.input or cfg["paths"]["raw_video"]
    clean = args.output or cfg["paths"]["clean_video"]

    if os.path.exists(raw):
        strip_metadata(raw, clean)
    else:
        print(f"[!] Raw video not found: {os.path.abspath(raw)}")
