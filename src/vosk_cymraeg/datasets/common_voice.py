import logging
import os
from pathlib import Path

import datasets
from dotenv import load_dotenv

from vosk_cymraeg.datasets.hf_utils import (
    create_combined_split,
    dump_dataset_audio_files,
)

_logger = logging.getLogger(__name__)

# Mapping from HF split names to our standard names
# Available splits: train, dev, test, train_all, train_missing, other_with_excluded
SPLIT_MAPPING = {
    "train": "train",
    "dev": "dev",
    "test": "test",
    "other_with_excluded": "other",
}


def fetch_common_voice(output_path: Path) -> None:
    """Fetches Common Voice Welsh dataset from HuggingFace techiaith/commonvoice_23_0_cy"""
    _logger.info("Loading dataset 'techiaith/commonvoice_23_0_cy' from HuggingFace")

    load_dotenv()
    token = os.environ.get("HF_TOKEN")

    dataset_splits = list(SPLIT_MAPPING.keys())

    for hf_split in dataset_splits:
        our_split = SPLIT_MAPPING[hf_split]
        _logger.info(f"Processing split: {hf_split} -> {our_split}")

        ds: datasets.Dataset = datasets.load_dataset(
            "techiaith/commonvoice_23_0_cy", split=hf_split, token=token
        )

        # Determine unique client ID length for speaker naming
        cid_length = _determine_cid_length(ds)
        _logger.info(f"Using client ID suffix length: {cid_length}")

        # Add speaker column (cvcy- prefix + last N chars of client_id)
        ds = ds.map(
            lambda row: {"speaker": f"cvcy-{row['client_id'][-cid_length:]}"},
            desc="Adding speaker IDs",
        )

        # Add utterance column (speaker + extracted number from path)
        ds = ds.map(
            lambda row: {
                "utterance": f"{row['speaker']}-{_extract_id(row['path'])}"
            },
            desc="Adding utterance IDs",
        )

        # Dump audio files and replace path column with new paths
        paths = dump_dataset_audio_files(ds, output_path)
        ds = ds.remove_columns("path")
        ds = ds.add_column("path", paths)
        ds = ds.add_column("lang", ["cy"] * len(ds))

        # Remove audio to save memory
        ds = ds.remove_columns("audio")

        # Write to CSV
        ds.to_csv(
            output_path / f"{our_split}.csv",
            columns=["speaker", "utterance", "path", "lang", "sentence"],
        )

    # Combine all splits into all.csv
    our_splits = list(SPLIT_MAPPING.values())
    all_df = create_combined_split(output_path, our_splits, validate=False)
    all_df.write_csv(output_path / "all.csv")


def _determine_cid_length(ds: datasets.Dataset) -> int:
    """Find the smallest suffix length that yields unique client IDs"""
    unique_clients = set(ds["client_id"])

    for length in range(4, 64):
        shortened = {cid[-length:] for cid in unique_clients}
        if len(shortened) == len(unique_clients):
            return length

    # Fallback to full length
    return 64


def _extract_id(path: str) -> str:
    """Extract numeric ID from path like 'common_voice_cy_12345.mp3'"""
    import re
    match = re.search(r"(\d+)", path)
    return match.group(1) if match else path
