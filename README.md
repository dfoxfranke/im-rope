# `im-rope`

This Rust crate provides a rope implementation based on RRB vectors.

Similarly to the standard library `String` type, a `Rope` owns its storage and
guarantees that its contents are valid Unicode. Unlike a `String`, it is backed
not by a `Vec<u8>` but by a `Vector<u8>` from the [`im`](https://docs.rs/im)
crate. These in turn are backed by a balanced tree structure known as an an RRB
tree, which makes a wide variety of operations asymptotically efficient. In
particular, ropes can be cloned in constant time, and can be split or
concatenated in logarithmic time.

# Documentation

See [API docs on docs.rs](https://docs.rs/im-rope).

## License

This project licensed under the [Apache License
2.0](https://spdx.org/licenses/Apache-2.0.html) with [LLVM
exception](https://spdx.org/licenses/LLVM-exception.html). Unless you explicitly
state otherwise, any contribution intentionally submitted for inclusion in
`im-rope` by you, shall be licensed as Apache 2.0 with LLVM exception,
without any additional terms or conditions.
