# Cotengra

Tensor network contraction path optimization library (Rust) with Wolfram LibraryLink exports.

## Build (native)

Run a release build for your host platform:

    cargo build --release

Artifacts (in target/<triple>/release/):
- Linux: libcotengra.so
- macOS: libcotengra.dylib
- Windows (MSVC): cotengra.dll

## Cross Compilation

Add targets with rustup, then build with --target.

Example (macOS arm64 from x86_64 host):

    rustup target add aarch64-apple-darwin
    cargo build --release --target aarch64-apple-darwin

### macOS Universal (x86_64 + arm64)

    rustup target add x86_64-apple-darwin aarch64-apple-darwin
    cargo build --release --target x86_64-apple-darwin
    cargo build --release --target aarch64-apple-darwin
    mkdir -p dist
    lipo -create \
      target/x86_64-apple-darwin/release/libcotengra.dylib \
      target/aarch64-apple-darwin/release/libcotengra.dylib \
      -output dist/libcotengra_universal.dylib

### Linux aarch64 from x86_64 host

    sudo apt-get update
    sudo apt-get install -y gcc-aarch64-linux-gnu
    rustup target add aarch64-unknown-linux-gnu
    mkdir -p .cargo
    printf "[target.aarch64-unknown-linux-gnu]\nlinker = \"aarch64-linux-gnu-gcc\"\n" >> .cargo/config.toml
    cargo build --release --target aarch64-unknown-linux-gnu

### Windows (MSVC)

On Windows host (GitHub Actions already covers this):

    rustup target add x86_64-pc-windows-msvc
    cargo build --release --target x86_64-pc-windows-msvc

### Optional: MUSL (more static Linux)

    rustup target add x86_64-unknown-linux-musl
    cargo build --release --target x86_64-unknown-linux-musl

## Continuous Integration

`.github/workflows/ci.yml` builds and uploads artifacts for:
- x86_64-unknown-linux-gnu
- aarch64-unknown-linux-gnu
- x86_64-apple-darwin
- aarch64-apple-darwin (merged via lipo into universal dylib)
- x86_64-pc-windows-msvc

## Wolfram Usage
## Packaging macOS Library into Paclet

After building the release library locally on macOS:

```
chmod +x scripts/package_macos.sh
./scripts/package_macos.sh
```

This copies `target/release/libcotengra.dylib` into the appropriate `Cotengra/LibraryResources/MacOSX-<arch>/` directory so the paclet can load it via `LibraryFunctionLoad` automatically.

For universal binaries, first lipo the two arch builds into `dist/libcotengra_universal.dylib` and then copy that instead (renaming to `libcotengra.dylib`).


Load the produced shared library with `LibraryFunctionLoad` to access the exported functions (annotated with `#[wll::export]`). Ensure Wolfram installation is discoverable for the build (override with `WOLFRAM_APPDIR` if necessary).
