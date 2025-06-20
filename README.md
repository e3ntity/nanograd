# nanograd

Minimal automatic-differentiation engine written in Rust.

## Quick start

```bash
cargo run --release
```

Running the demo trains a tiny Multi-Layer Perceptron to learn the XOR function and writes a rendered computation graph of a single Perceptron to `graph.html`.

## Usage

```rust
use nanograd::scalar::{Scalar, func};

let x = Scalar::new_grad(2.0);
let y = Scalar::new_grad(-1.0);

// z = relu(x * y + 3)
let z = func::relu(x.clone() * y.clone() + 3.0);
z.backward();

println!("z = {}", z.get_value());
println!("dz/dx = {:?}", x.get_grad());
println!("dz/dy = {:?}", y.get_grad());
```

## How it works

- Each `Scalar` stores **value**, optional **gradient**, and an `Edge` describing the operation that produced it.
- Operator overloads (`+`, `*`, `-`, ...) and helpers in `scalar::func` build a directed acyclic graph of `Scalar`s while caching the **local derivative** for every edge.
- `backward()` starts at an output node and **recursively propagates gradients** along the graph, accumulating them into the leaf nodes created with `Scalar::new_grad`.
- The graph can be visualised with `plot::dump_graph`, which emits a self-contained D3.js HTML file.
