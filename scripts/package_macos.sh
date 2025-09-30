#!/usr/bin/env bash
set -euo pipefail

# Determine architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
  DEST_DIR="Cotengra/LibraryResources/MacOSX-ARM64"
else
  DEST_DIR="Cotengra/LibraryResources/MacOSX-x86-64"
fi

LIB_SRC="target/release/libcotengra.dylib"
if [[ ! -f "$LIB_SRC" ]]; then
  echo "Release macOS dylib not found at $LIB_SRC. Building..." >&2
  cargo build --release
fi
mkdir -p "$DEST_DIR"
cp "$LIB_SRC" "$DEST_DIR/"

echo "Copied libcotengra.dylib to $DEST_DIR"