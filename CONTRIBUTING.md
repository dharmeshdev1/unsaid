# Contributing

Bug reports, measurements, new clients and documentation fixes are all welcome. For something large,
please open an issue first so that we can agree on the approach. [Ideas to build on this](docs/project-ideas.md)
has plenty of starting points.

## Setting up

You need a stable Rust toolchain. Everything builds with `cargo`:

```sh
cargo test --workspace --release
cargo clippy --workspace --all-targets
cargo fmt --all --check
```

The web interface needs a little more:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.106 --locked
sh scripts/build_web.sh
uv run --with playwright scripts/test_web.py http://127.0.0.1:8377
```

## Ground rules

- `web/pkg` is committed, so that `cargo install` works without WebAssembly tools. Rebuild it whenever
  the client changes.
- Any number in the documentation must come from a command that someone else can run.
- Code handling server data must assume that the server is hostile. Return errors rather than
  panicking, and bound every allocation.
- `unsafe` is forbidden, except in the glue that `wasm-bindgen` generates.
- A change to the wire format must update [`docs/protocol.md`](docs/protocol.md) and the Python client
  in [`tutorial/3_real_client.py`](tutorial/3_real_client.py).

Contributions are accepted under the Apache License 2.0, the same license as the project.
