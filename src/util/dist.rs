use rand::Rng;

pub fn rand_normal() -> f64 {
    let mut rng = rand::rng();

    let u1: f64 = rng.random_range(0.0..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);

    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
