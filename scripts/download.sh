#!/bin/bash

set -euo pipefail

OUT_DIR="./dataset"
mkdir -p "$OUT_DIR"

URLS=(
"https://zenodo.org/records/17943574/files/Part_1.zip?download=1"
"https://zenodo.org/records/17943574/files/Part_2.zip?download=1"
"https://zenodo.org/records/17943574/files/Part_3.zip?download=1"
"https://zenodo.org/records/17943574/files/Part_4.zip?download=1"
"https://zenodo.org/records/17943574/files/Part_5.zip?download=1"
"https://zenodo.org/records/17943574/files/Part_6.zip?download=1"
"https://zenodo.org/records/17943574/files/Part_7.zip?download=1"
)

for i in "${!URLS[@]}"
do
part=$((i+1))
filename="Part_${part}.zip"
filepath="$OUT_DIR/$filename"
extracted_dir="$OUT_DIR/split_${part}"

echo
echo "=================================================="
echo "Processing $filename"
echo "=================================================="

if [ -d "$extracted_dir" ]; then
    echo "[$(date)] $extracted_dir already exists. Skip processing."
    continue
fi

if [ -f "$filepath" ]; then
    echo "[$(date)] $filename already exists. Skip download."
else
    echo "[$(date)] Downloading $filename"

    wget -c -O "$filepath" "${URLS[$i]}"
fi

echo "[$(date)] Extracting $filename"

unzip -o "$filepath" -d "$OUT_DIR"

echo "[$(date)] Removing $filename"

rm -f "$filepath"

echo "[$(date)] Completed $filename"

done

echo
echo "[$(date)] ALL DONE"
