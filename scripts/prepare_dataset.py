#!/usr/bin/env python3
import json
import os
import shutil
import csv
from pathlib import Path
import sys

# Add root folder to sys.path to import from merge_splits
sys.path.append(str(Path(__file__).resolve().parents[1]))
from merge_splits import merge_dataset

def main():
    root_dir = Path(__file__).resolve().parents[1]
    splits_root = root_dir / "dataset"
    merged_dir = root_dir / "merged"
    processed_dir = root_dir / "data" / "processed" / "vsl_400"

    print("=== Step 1: Merging splits from ./dataset to ./merged ===")
    if merged_dir.exists():
        print(f"Directory {merged_dir} already exists. Skipping merge. If you want to re-merge, delete it first.")
    else:
        try:
            merge_dataset(
                root=splits_root,
                out_dir=merged_dir,
                copy_mode="hardlink",
                overwrite=True,
                stop_on_dup=False
            )
            print("Merge completed successfully.")
        except Exception as e:
            print(f"Error during merge: {e}")
            sys.exit(1)

    print("\n=== Step 2: Preparing dataset directory structure under data/processed/vsl_400 ===")
    processed_dir.mkdir(parents=True, exist_ok=True)

    view_mapping = {
        "front_view": "cam_1",
        "left_view": "cam_2",
        "right_view": "cam_3"
    }

    for src_name, dst_name in view_mapping.items():
        # Handle video directory
        src_video_dir = merged_dir / src_name
        dst_video_dir = processed_dir / dst_name
        if src_video_dir.exists():
            if not dst_video_dir.exists():
                print(f"Creating symlink/copy: {src_name} -> {dst_name}")
                try:
                    # Try creating symlink first using os.path.relpath
                    rel_path = os.path.relpath(src_video_dir, dst_video_dir.parent)
                    dst_video_dir.symlink_to(rel_path, target_is_directory=True)
                except Exception:
                    # Fallback to copying/linking files if symlink fails
                    shutil.copytree(src_video_dir, dst_video_dir)
            else:
                print(f"Directory {dst_video_dir} already exists.")
        else:
            print(f"Warning: {src_video_dir} does not exist.")

        # Handle metadata JSON file
        src_json = merged_dir / f"{src_name}.json"
        dst_json = processed_dir / f"{dst_name}.json"
        if src_json.exists():
            if not dst_json.exists():
                print(f"Linking metadata: {src_name}.json -> {dst_name}.json")
                try:
                    rel_path = os.path.relpath(src_json, dst_json.parent)
                    dst_json.symlink_to(rel_path)
                except Exception:
                    shutil.copy2(src_json, dst_json)
            else:
                print(f"File {dst_json} already exists.")
        else:
            print(f"Warning: {src_json} does not exist.")

    print("\n=== Step 3: Generating gloss.csv mapping file ===")
    gloss_csv_path = processed_dir / "gloss.csv"
    if gloss_csv_path.exists():
        print(f"gloss.csv already exists at {gloss_csv_path}")
    else:
        # Load unique glosses from one of the merged json files
        sample_json = processed_dir / "cam_1.json"
        if not sample_json.exists():
            # Fallback to checking in merged folder
            sample_json = merged_dir / "front_view.json"

        if sample_json.exists():
            print(f"Extracting glosses from {sample_json}...")
            with open(sample_json, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            unique_glosses = sorted(list(set(item["gloss"] for item in metadata)))
            print(f"Found {len(unique_glosses)} unique glosses.")
            
            # Write to CSV in id,gloss format
            with open(gloss_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                for idx, gloss in enumerate(unique_glosses):
                    writer.writerow([idx, gloss])
            print(f"Successfully generated {gloss_csv_path}")
        else:
            print("Error: Could not find cam_1.json or front_view.json to extract glosses.")
            sys.exit(1)

    print("\nPreparation completed successfully! You can now start training.")

if __name__ == "__main__":
    main()
