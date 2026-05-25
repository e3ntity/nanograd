use core::fmt;
use std::{
    fs::File,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
};

use crate::examples::mnist::{dataset::load_image, network::Network};

const INDEX_HTML: &str = include_str!("index.html");

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[derive(Debug)]
pub enum HttpMethod {
    Get,
    Post,
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            HttpMethod::Get => write!(f, "GET"),
            HttpMethod::Post => write!(f, "POST"),
        }
    }
}

pub struct Request {
    data: Vec<u8>,
}

impl Request {
    pub fn new() -> Self {
        let data = vec![];

        Request { data }
    }

    pub fn extend(&mut self, data: &[u8]) -> () {
        self.data.extend_from_slice(data);
    }

    pub fn head(&self) -> Option<Vec<u8>> {
        let Some(split) = find(&self.data, b"\r\n\r\n") else {
            return None;
        };

        return Some(self.data[0..split].to_vec());
    }

    pub fn body(&self) -> Option<Vec<u8>> {
        let Some(split) = find(&self.data, b"\r\n\r\n") else {
            return None;
        };

        return Some(self.data[split + 4..].to_vec());
    }

    pub fn header(&self, key: &str) -> Option<String> {
        let Some(head) = self.head() else {
            return None;
        };

        let prefix = format!("{}: ", key.to_ascii_lowercase());
        for header in String::from_utf8_lossy(&*head).lines() {
            let header_ = header.to_ascii_lowercase();
            let Some(value) = header_.strip_prefix(&*prefix) else {
                continue;
            };

            return Some(value.trim().to_owned());
        }

        None
    }

    pub fn method(&self) -> Option<HttpMethod> {
        let Some(head_raw) = self.head() else {
            return None;
        };
        let head = String::from_utf8_lossy(&*head_raw).to_ascii_lowercase();

        let Some(method) = head.split_whitespace().nth(0) else {
            return None;
        };
        match method {
            "get" => Some(HttpMethod::Get),
            "post" => Some(HttpMethod::Post),
            _ => None,
        }
    }

    pub fn path(&self) -> Option<String> {
        let Some(head_raw) = self.head() else {
            return None;
        };
        let head = String::from_utf8_lossy(&*head_raw);

        let Some(path) = head.split_whitespace().nth(1) else {
            return None;
        };

        Some(path.to_owned())
    }

    pub fn route(&self) -> Option<String> {
        let Some(path) = self.path() else {
            return None;
        };

        let Some(route) = path.split("?").next() else {
            return None;
        };

        Some(route.to_owned())
    }

    pub fn header_len(&self) -> usize {
        self.head().unwrap_or(vec![]).len()
    }

    pub fn body_len(&self) -> usize {
        return self.len() - self.header_len() - 4;
    }

    pub fn len(&self) -> usize {
        return self.data.len();
    }
}

pub fn handle_connection(mut stream: TcpStream, model: Network) {
    let mut req = Request::new();
    let mut buf = [0u8; 8192];

    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                req.extend(&buf[..n]);

                if req.head().is_none() {
                    continue;
                }
                let content_length: usize = req
                    .header("Content-Length")
                    .unwrap_or("0".to_owned())
                    .trim()
                    .parse()
                    .unwrap_or(0);
                if req.body_len() >= content_length {
                    break;
                }
            }
            Err(_) => return,
        }
    }

    let Some(route) = req.route() else {
        let _ = stream.write_all(format!("HTTP/1.1 500 Internal Server Error\r\n\r\n").as_bytes());
        return;
    };
    let Some(method) = req.method() else {
        let _ = stream.write_all(format!("HTTP/1.1 500 Internal Server Error\r\n\r\n").as_bytes());
        return;
    };

    if route.to_ascii_lowercase() != "/" {
        let _ = stream.write_all(format!("HTTP/1.1 404 Not Found\r\n\r\n").as_bytes());
        return;
    }
    match method {
        HttpMethod::Get => {
            let body = INDEX_HTML;
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 200 Ok\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                )
                .as_bytes(),
            );
            return;
        }
        HttpMethod::Post => {}
    }

    let content_type = req
        .header("Content-Type")
        .unwrap_or("".to_owned())
        .to_ascii_lowercase();
    if content_type != "image/jpeg" {
        let _ = stream.write_all(format!("HTTP/1.1 400 Bad Request\r\n\r\n").as_bytes());
        return;
    }

    let id: u64 = rand::random();
    let path = format!("/tmp/nanograd-mnist-{id}.jpg");
    let body = req.body().unwrap();
    if File::create(path.clone())
        .and_then(|mut f| f.write_all(&*body))
        .is_err()
    {
        let _ = stream.write_all(format!("HTTP/1.1 500 Internal Server Error\r\n\r\n").as_bytes());
        return;
    }

    let Ok(image) = load_image(path) else {
        let _ = stream.write_all(format!("HTTP/1.1 500 Internal Server Error\r\n\r\n").as_bytes());
        return;
    };
    let predictions = model.forward(image);

    let body = format!(
        "{{\"predictions\": [{}]}}",
        predictions.map(|s| s.get_value().to_string()).join(",")
    );

    let _ = stream.write_all(
        format!(
            "HTTP/1.1 200 Ok\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        )
        .as_bytes(),
    );
}
