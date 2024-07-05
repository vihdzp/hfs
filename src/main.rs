//! # Hereditarily finite sets

#![warn(clippy::pedantic)]
#![warn(missing_docs)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_docs_in_private_items)]

mod mset;

pub mod prelude;
pub mod set;
pub mod utils;
use prelude::*;

/// Small vector.
type SmallVec<T> = smallvec::SmallVec<[T; 8]>;

/// [`smallvec::smallvec`] coerced into [`SmallVec`].
#[macro_export]
macro_rules! smallvec {
    ($elem: expr; $n: expr) => (
        SmallVec::from_elem($elem, $n)
    );
    ($($x: expr), *$(,)*) => ({
        let vec: SmallVec<_> = smallvec::smallvec![$($x,)*];
        vec
    });
}

fn main() {
    let a = Set::nat(2);
    let b = Set::nat(3);
    dbg!(b.rank());
    let c = a.clone().inter(b.clone());

    println!("A:     {a}\nB:     {b}\nA ∩ B: {c}");
}
