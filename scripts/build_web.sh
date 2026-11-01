#!/bin/sh
# Rebuild the browser client into web/pkg, which the `unsaid` binary embeds.
# Needs the wasm32-unknown-unknown target and wasm-bindgen-cli 0.2.106.
set -eu
cd "$(dirname "$0")/.."
cargo build --release -p unsaid-wasm --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/unsaid_wasm.wasm
ls -l web/pkg
