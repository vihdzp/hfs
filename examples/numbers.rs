//! The standard set-theoretic construction for the integers ℤ consists of equivalence classes in
//! ℕ², and likewise, rationals ℚ are formed from equivalence classes in ℤ × ℕ⁺. These are
//! unfortunately infinite sets, so we're forced to come up with alternate ways to model these
//! collections.
//!
//! By doing this, we can then model real numbers as classes of Dedekind cuts in ℚ.

#![allow(dead_code)]

use hfs::prelude::*;

/// We model integers as something resembling a natural with a sign bit. For n > 0:
///
/// - +n = (0, n)
/// - &pm;0 = (0, 0)
/// - &minus;n = (1, n)
///
/// Try changing `nat` for `zermelo` for more readable output.
fn int(n: isize) -> Set {
    // Safety: both elements of our pairs are distinct.
    unsafe {
        match n {
            0 => Set::nat(0).id_kpair(),
            -1 => Set::nat(1).id_kpair(),
            1.. => Set::nat(0).kpair_unchecked(Set::nat(n as usize)),
            ..=-2 => Set::nat(1).kpair_unchecked(Set::nat((-n) as usize)),
        }
    }
}

/// Defines a bijection between naturals and integers.
fn nat_to_int(n: usize) -> isize {
    let x = ((n + 1) / 2) as isize;
    if n % 2 == 0 {
        x
    } else {
        -x
    }
}

/// The class of integers.
fn class_int() -> Class {
    // Safety: this iterator has no duplicates.
    unsafe { Class::new_unchecked((0..).map(|n| int(nat_to_int(n)))) }
}

/// Pairs up an integer and a natural number.
fn rat_unchecked(m: isize, n: usize) -> Set {
    // Safety: no natural number is a valid Kuratowski pair.
    // Exercise: what integers would correspond to natural numbers under the Zermelo encoding?
    unsafe { int(m).kpair_unchecked(Set::nat(n)) }
}

/// A rational m&frasl;n is a pair of an integer and a natural number, representing the fraction in
/// lowest form.
///
/// Returns `None` if `n == 0`.
fn rat(m: isize, n: usize) -> Option<Set> {
    /// Reduces a fraction into lowest terms.
    fn reduce(m: usize, n: usize) -> (usize, usize) {
        let g = gcd::binary_usize(m, n);
        (m / g, n / g)
    }

    // Division by zero is invalid.
    if n == 0 {
        return None;
    }

    // Reduce fraction.
    let (m, n) = if m >= 0 {
        let (x, y) = reduce(m as usize, n);
        (x as isize, y)
    } else {
        let (x, y) = reduce((-m) as usize, n);
        (-(x as isize), y)
    };

    Some(rat_unchecked(m, n))
}

/// The class of rationals.
fn class_rat() -> Class {
    // Safety: this iterator has no duplicates.
    unsafe {
        Class::new_unchecked(class::Product::new(0.., 1..).filter_map(|(m, n)| {
            let m = nat_to_int(m);
            if gcd::binary_usize(m.unsigned_abs(), n) == 1 {
                Some(rat_unchecked(m, n))
            } else {
                None
            }
        }))
    }
}

/// Encodes a "real" number as a Dedekind cut.
///
/// Of course, this ignores that floating points are all rational numbers to begin with. A more
/// faithful construction might instead allow for custom selection predicates, subject to the usual
/// conditions for a Dedekind cut.
fn real(x: f64) -> Class {
    class_rat().select(move |r| {
        // Retrieve rational.
        // Safety: these are guaranteed to be valid Kuratowski pairs.
        let r = unsafe {
            let (k, n) = r.ksplit().unwrap_unchecked().pair();
            let (s, m) = k.ksplit().unwrap_unchecked().pair();
            let f = m.card() as f64 / n.card() as f64;
            if s.is_empty() {
                f
            } else {
                -f
            }
        };

        r < x
    })
}

/// Print out the class representing √2.
fn main() {
    println!("Square root of 2:\n{{");
    for set in real(f64::sqrt(2.0)).into_iter().take(10) {
        println!("    {set},");
    }
    println!("    ...\n}}");
}
