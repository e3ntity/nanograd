mod error;
mod examples;
mod nn;
mod scalar;
mod tensor;
mod util;

fn usage() -> ! {
    eprintln!("Usage: nanograd <example>");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  xor                 Train an MLP on XOR and dump a perceptron graph");
    eprintln!("  mnist serve [file]  Serve the trained MNIST web demo (default: mnist.ng)");
    eprintln!("  mnist train [file]  Train the MNIST CNN and save it (default: mnist.ng)");
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("xor") => examples::xor::run(),
        Some("mnist") => {
            let file = args.get(3).cloned().unwrap_or_else(|| "mnist.ng".to_owned());
            match args.get(2).map(String::as_str) {
                Some("serve") => examples::mnist::serve(file),
                Some("train") => examples::mnist::train(file),
                _ => usage(),
            }
        }
        _ => usage(),
    }
}
