//! Scalar type and automatic differentiation primitives.
//!
//! The `scalar` module models a simple reverse-mode automatic differentiation
//! engine for scalar values.  Each [`Scalar`] holds a value, an optional
//! gradient and a link (an *edge*) to the operation that produced it.  Calling
//! [`Scalar::backward`] traverses the computational graph in reverse order and
//! accumulates gradients on every node that was created with gradient tracking
//! enabled.
use std::{
    ops,
    sync::{Arc, Mutex},
};

#[derive(Debug)]
pub(super) enum Operation {
    Abs,
    Add,
    Ln,
    Max,
    Mul,
    Neg,
    Relu,
    Sigmoid,
    Sqrt,
    Sub,
}

/// Edge connecting an operation with its input nodes.  Each tuple in
/// `operands` stores a reference to the child node together with its local
/// derivative (∂out/∂child).
#[derive(Debug)]
pub(super) struct Edge {
    pub(crate) operands: Vec<(Arc<Node>, f32)>,
    pub(crate) operation: Operation,
}

impl Edge {
    pub fn new(operands: Vec<(Arc<Node>, f32)>, operation: Operation) -> Self {
        Edge {
            operands,
            operation,
        }
    }
}

/// Internal node that backs a public [`Scalar`].  It is shared via `Arc` so
/// that several `Scalar`s can point to the same allocation.
#[derive(Debug)]
pub(super) struct Node {
    pub(super) edge: Arc<Mutex<Option<Edge>>>,
    pub(super) grad: Mutex<Option<f32>>,
    pub(super) value: Mutex<f32>,
}

/// Public handle to a scalar value that participates in automatic
/// differentiation.
///
/// Use [`Scalar::new`] for plain numeric values and [`Scalar::new_grad`] when
/// the scalar should accumulate gradients (i.e. is a *leaf* in the graph).
#[derive(Clone, Debug)]
pub struct Scalar(Arc<Node>);

impl Scalar {
    /// Construct a scalar that does **not** accumulate gradients.
    pub fn new(value: f32) -> Self {
        Self::new_(value, None)
    }

    /// Construct a scalar that will accumulate gradients.  The initial
    /// gradient is zero.
    pub fn new_grad(value: f32) -> Self {
        Self::new_(value, Some(0.))
    }

    fn new_(value: f32, grad: Option<f32>) -> Self {
        Scalar(Arc::new(Node {
            edge: Arc::new(Mutex::new(None)),
            grad: Mutex::new(grad),
            value: Mutex::new(value),
        }))
    }

    /// Entry-point for reverse-mode AD.  Starts a backward pass with
    /// the seed gradient `1.0`.
    pub fn backward(&self) {
        if self.0.grad.lock().unwrap().is_none() {
            panic!("Cannot call backward on Scalar without gradient enabled");
        }

        self.backward_(1.);
    }

    /// Same as [`Scalar::backward`] but allows passing an explicit seed value.
    pub fn backward_(&self, val: f32) {
        self.set_grad(self.get_grad().unwrap() + val);

        let clone = Arc::clone(&self.0.edge);
        let lock = clone.lock().unwrap();
        let Some(edge) = &*lock else {
            return;
        };

        for (child, weight) in edge.operands.iter() {
            let child_scalar = Scalar(child.clone());
            if child_scalar.has_grad() {
                child_scalar.backward_(val * weight);
            }
        }
    }

    /// Returns `true` if a gradient buffer is allocated for this scalar.
    pub fn has_grad(&self) -> bool {
        self.0.grad.lock().unwrap().is_some()
    }

    /// Current value of the gradient buffer, if any.
    pub fn get_grad(&self) -> Option<f32> {
        *self.0.grad.lock().unwrap()
    }

    /// Overwrite the gradient buffer with `val`.  Creates the buffer if it
    /// did not exist before.
    pub fn set_grad(&self, val: f32) {
        *self.0.grad.lock().unwrap() = Some(val);
    }

    /// Numerical value stored in the scalar.
    pub fn get_value(&self) -> f32 {
        *self.0.value.lock().unwrap()
    }

    /// Mutably set the stored numerical value.
    pub fn set_value(&self, value: f32) {
        *self.0.value.lock().unwrap() = value;
    }

    /// Attach an `edge` describing how this scalar was obtained from its
    /// operands.  This is called by the arithmetic operator overloads.
    pub(super) fn set_edge(&self, edge: Edge) {
        for (child, _) in edge.operands.iter() {
            if Scalar(child.clone()).has_grad() {
                self.set_grad(0.);
                break;
            }
        }

        *self.0.edge.lock().unwrap() = Some(edge);
    }

    /// Immediate children of this node together with their local derivatives.
    pub fn get_children(&self) -> Vec<(Scalar, f32)> {
        let clone = Arc::clone(&self.0.edge);
        let lock = clone.lock().unwrap();
        let Some(edge) = &*lock else {
            return Vec::new();
        };

        edge.operands
            .iter()
            .map(|c| (Scalar(c.0.clone()), c.1))
            .collect()
    }

    /// Raw pointer address of the shared node.  Useful as a unique identifier
    /// when exporting or visualising the graph.
    pub fn get_arc_ptr(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    /// Human-readable name of the operation that produced this scalar.  Returns
    /// `"Input"` for leaf nodes.
    pub fn get_operation_name(&self) -> String {
        let clone = Arc::clone(&self.0.edge);
        let lock = clone.lock().unwrap();
        let Some(edge) = &*lock else {
            return "Input".to_owned();
        };

        match edge.operation {
            Operation::Add => "Add".to_owned(),
            Operation::Ln => "Ln".to_owned(),
            Operation::Max => "Max".to_owned(),
            Operation::Mul => "Mul".to_owned(),
            Operation::Neg => "Neg".to_owned(),
            Operation::Relu => "ReLU".to_owned(),
            Operation::Sigmoid => "Sigmoid".to_owned(),
            Operation::Sqrt => "Sqrt".to_owned(),
            Operation::Sub => "Sub".to_owned(),
            Operation::Abs => "Abs".to_owned(),
        }
    }
}

impl Into<Arc<Node>> for Scalar {
    fn into(self) -> Arc<Node> {
        self.0
    }
}

impl Into<Scalar> for Arc<Node> {
    fn into(self) -> Scalar {
        Scalar(self)
    }
}

impl ops::Add<Scalar> for Scalar {
    type Output = Self;

    fn add(self, rhs: Scalar) -> Self::Output {
        let out = Scalar::new(self.get_value() + rhs.get_value());
        out.set_edge(Edge::new(
            vec![(self.clone().into(), 1.), (rhs.clone().into(), 1.)],
            Operation::Add,
        ));

        out
    }
}

impl ops::Add<f32> for Scalar {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        self + Scalar::new(rhs)
    }
}

impl ops::Add<Scalar> for f32 {
    type Output = Scalar;

    fn add(self, rhs: Scalar) -> Self::Output {
        rhs + self
    }
}

impl ops::Sub<Scalar> for Scalar {
    type Output = Self;

    fn sub(self, rhs: Scalar) -> Self::Output {
        self + -rhs
    }
}

impl ops::Sub<f32> for Scalar {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        self - Scalar::new(rhs)
    }
}

impl ops::Sub<Scalar> for f32 {
    type Output = Scalar;

    fn sub(self, rhs: Scalar) -> Self::Output {
        -rhs + self
    }
}

impl ops::Mul<Scalar> for Scalar {
    type Output = Self;

    fn mul(self, rhs: Scalar) -> Self::Output {
        let out = Scalar::new(self.get_value() * rhs.get_value());
        out.set_edge(Edge::new(
            vec![
                (self.clone().into(), rhs.get_value()),
                (rhs.clone().into(), self.get_value()),
            ],
            Operation::Mul,
        ));

        out
    }
}

impl ops::Mul<f32> for Scalar {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        self * Scalar::new(rhs)
    }
}

impl ops::Mul<Scalar> for f32 {
    type Output = Scalar;

    fn mul(self, rhs: Scalar) -> Self::Output {
        rhs * self
    }
}

impl ops::Neg for Scalar {
    type Output = Scalar;

    fn neg(self) -> Self::Output {
        let out = Scalar::new(-self.get_value());
        out.set_edge(Edge::new(vec![(self.clone().into(), -1.)], Operation::Neg));

        out
    }
}
