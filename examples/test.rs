//! I'm just using this to test out some code.

use hfs::prelude::*;

fn main() {
    for pair in Set::nat(2).tag_union(Set::nat(3)) {
        let (x, y) = pair.into_ksplit().unwrap();
        println!("{}, {}", x.card(), y.card())
    }
}
