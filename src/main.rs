//! # Hereditarily finite sets

#![warn(clippy::pedantic)]

mod mset;

mod prelude;
mod set;
mod utils;
use prelude::*;

/// Small vector.
type SmallVec<T> = smallvec::SmallVec<[T; 8]>;

fn main() {
    let a: Mset = "{{}, {}, {{}, {}}, {{}}}".parse().unwrap();
    let b: Set = a.clone().into_set();

    println!("Set A: {a}\nSet B: {b}");
}
