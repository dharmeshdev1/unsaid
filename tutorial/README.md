# Private search from scratch

You can understand this entire system in an afternoon. Each file runs on its own and needs only
[numpy](https://numpy.org). No background in cryptography is assumed, only some linear algebra.

| File | What you will learn | Time |
| --- | --- | --- |
| [`1_lwe_by_hand.py`](1_lwe_by_hand.py) | Why noisy equations are hard, how to hide a message under them, and how a server computes on what it cannot read | 15 minutes |
| [`2_pir_from_scratch.py`](2_pir_from_scratch.py) | Fetching a record and ranking documents privately, with both sides in one file | 20 minutes |
| [`3_real_client.py`](3_real_client.py) | A complete client in about 250 lines of Python, working against the real Rust server | An hour |

```sh
pip install numpy
python3 tutorial/1_lwe_by_hand.py
python3 tutorial/2_pir_from_scratch.py

unsaid demo                                            # in another terminal
python3 tutorial/3_real_client.py "how do vaccines work"
```

After that, read [how it works](../docs/how-it-works.md) and the [protocol](../docs/protocol.md).

## Things to try

These are the best way to find out whether you really understood it.

1. In part 1, raise the noise until decryption starts to fail. Work out the limit on paper first.
2. In part 2, use the same secret for two different queries. Subtract the ciphertexts and see what the
   server learns. This is why the Rust type system refuses to let it happen.
3. In part 3, log every request on the server. Search for two very different things and try to tell
   the logs apart.
4. In part 3, send the server a wrong answer on purpose and watch the page hash catch it. Then ask
   what the server could have learned from what the client did next.
