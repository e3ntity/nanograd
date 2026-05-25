use std::sync::Arc;

use crate::scalar::scalar::Node;

use super::Scalar;
use super::scalar::Edge;
use super::scalar::Operation;

pub fn abs(x: Scalar) -> Scalar {
    let v = x.get_value();
    let value = if v > 0. { v } else { -v };

    let out = Scalar::new(value);
    let weight = if v > 0. {
        1.
    } else if v == 0. {
        0.
    } else {
        -1.
    };
    out.set_edge(Edge::new(vec![(x.clone().into(), weight)], Operation::Abs));

    out
}

pub fn max(x: Scalar, y: Scalar) -> Scalar {
    let xv = x.get_value();
    let yv = y.get_value();
    let value = if xv > yv { xv } else { yv };

    let out = Scalar::new(value);
    let node = if xv > yv { x.clone() } else { y.clone() };
    out.set_edge(Edge::new(vec![(node.into(), 1.)], Operation::Max));

    out
}

pub fn ln(x: Scalar) -> Scalar {
    let v = x.get_value();

    let out = Scalar::new(v.ln());
    let weight = 1. / v;
    out.set_edge(Edge::new(vec![(x.clone().into(), weight)], Operation::Ln));

    out
}

pub fn exp(x: Scalar) -> Scalar {
    let v = x.get_value();

    let out = Scalar::new(v.exp());
    let weight = v.exp();
    out.set_edge(Edge::new(vec![(x.clone().into(), weight)], Operation::Exp));

    out
}

pub fn sqrt(x: Scalar) -> Scalar {
    let v = x.get_value();

    let out = Scalar::new(v.sqrt());
    let weight = 1. / (2. * v.sqrt());
    out.set_edge(Edge::new(vec![(x.clone().into(), weight)], Operation::Sqrt));

    out
}

pub fn norm2(x: Scalar) -> Scalar {
    sqrt(x.clone() * x)
}

fn sigmoid_(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

pub fn sigmoid(x: Scalar) -> Scalar {
    let v = x.get_value();

    let out = Scalar::new(sigmoid_(v));
    let weight = sigmoid_(v) * (1. - sigmoid_(v));
    out.set_edge(Edge::new(
        vec![(x.clone().into(), weight)],
        Operation::Sigmoid,
    ));

    out
}

pub fn relu(x: Scalar) -> Scalar {
    let v = x.get_value();
    let value = if v > 0. { v } else { 0. };

    let out = Scalar::new(value);
    let weight = if value > 0. { 1. } else { 0. };
    out.set_edge(Edge::new(vec![(x.clone().into(), weight)], Operation::Relu));

    out
}

pub fn softmax<const N: usize>(x: [Scalar; N]) -> [Scalar; N] {
    let v: [f32; N] = std::array::from_fn(|i| x[i].get_value());
    let norm: f32 = v.iter().map(|v| v.exp()).sum();

    let out = std::array::from_fn(|i| Scalar::new(v[i].exp() / norm));
    for i in 0..N {
        let mut edges: Vec<(Arc<Node>, f32)> = vec![];
        for k in 0..N {
            let kd = if i == k { 1.0 } else { 0.0 };
            let weight = out[i].get_value() * (kd - out[k].get_value());
            edges.push((x[k].clone().into(), weight));
        }

        out[i].set_edge(Edge::new(edges, Operation::Softmax));
    }

    out
}
