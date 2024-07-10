//! Kuratowski's definition of ordered pairs (x, y) = {{x}, {x, y}} is pretty much universal within
//! foundational mathematics. However, other constructions are possible. Here we define a
//! "[shorter](https://en.wikipedia.org/wiki/Ordered_pair#Variants)" definition (x, y) = {x, {x,
//! y}}. The main disadvantages of this definition are that it does not work in the absence of the
//! axiom of regularity, and that splitting a set is thus significantly more involved.

#![allow(dead_code)]

use hfs::prelude::*;

/// Short ordered pair (x, y) = {x, {x, y}}.
fn spair(x: Set, y: Set) -> Set {
    // Safety: x ≠ {x, y} as they have distinct set rank.
    unsafe { x.clone().singleton().insert_unchecked(x.pair(y)) }
}

/// Splits a short pair.
fn ssplit(x: &Set) -> Option<(&Set, &Set)> {
    // Test whether fst = x and snd = {x, y} for some x, y.
    fn test<'a>(fst: &'a Set, snd: &'a Set) -> Option<(&'a Set, &'a Set)> {
        match snd.as_slice() {
            [a] => {
                if fst == a {
                    Some((a, a))
                } else {
                    None
                }
            }
            [a, b] => {
                if fst == a {
                    Some((a, b))
                } else if fst == b {
                    Some((b, a))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    match x.as_slice() {
        [fst, snd] => test(fst, snd).or_else(|| test(snd, fst)),
        _ => None,
    }
}

/// A fun coincidence: 2 = (0, 0).
fn main() {
    let nat = Set::nat(2);
    let (a, b) = ssplit(&nat).unwrap();
    println!("2 = ({a}, {b})");
}
