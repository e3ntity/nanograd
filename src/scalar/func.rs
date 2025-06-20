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
