//! General library tests.
//!
//! We maintain a [`Suite`] of generic sets for which we test our operations. Unfortunately, due to
//! how various tests' results have to be hardcoded, it's a bit difficult to add new sets to it.

#![cfg(test)]

use crate::prelude::*;

/// Creates analogous tests for [`Set`] and [`Mset`].
macro_rules! test {
    ($($name: ident),*) => {
        $(
            concat_idents::concat_idents!(fn_name = mset, $name {
                #[test]
                #[doc = concat!("Test [`Mset::", stringify!($name), "`].")]
                fn fn_name() {
                    Mset::$name();
                }
            });

            concat_idents::concat_idents!(fn_name = set, $name {
                #[test]
                #[doc = concat!("Test [`Set::", stringify!($name), "`].")]
                fn fn_name() {
                    Set::$name();
                }
            });
        )*
    };
}

/// Asserts that two booleans are equal. If they aren't, a string equal to either a whitespace or `"
/// not "` is passed to the formatting string, depending on the expected result.
///
/// ```
/// # let (expect, cmp) = (true, true);
/// // If `expect` and `!cmp`, this fails with "cmp is not true".
/// // If `!expect` and `cmp`, this fails with "cmp is true".
/// assert_cmp!(expect, cmp, "cmp is{}true");
/// ```
macro_rules! assert_beq {
    ($expect: expr, $cmp: expr, $msg: literal) => {
        let expect = $expect;
        if expect != $cmp {
            let not = if expect { " not " } else { " " };
            panic!($msg, not);
        }
    };
}

/// Our suite of sets to perform tests on.
trait Suite: SetTrait {
    /// A multitude of general sets for general-purpose testing.
    ///
    /// Both the list and each constituent set should be normalized, i.e. in decreasing
    /// lexicographic order.
    const SUITE: &'static [&'static str];

    /// Hardcoded indices of pairs of sets containing each other.
    const MEM: &'static [(usize, usize)];

    /// Parses string or panics.
    fn parse(str: &str) -> Self {
        str.parse().expect("could not parse string")
    }

    /// Normalize string for a set.
    #[must_use]
    fn normalize(str: &str) -> String {
        Self::parse(str).to_string()
    }

    /// Verify round-trip conversion between a set and a string.
    fn roundtrip(&self, str: &str) {
        assert_eq!(
            self,
            &Self::parse(str),
            "roundtrip fail: {str} not mapped to {self}"
        );
        assert_eq!(
            self.to_string(),
            str,
            "roundtrip fail: {self} not mapped to {str}"
        );
    }

    /// Our [`SUITE`](Suite::SUITE) as `(usize, &str, Self)` tuples.
    fn suite() -> impl Iterator<Item = (usize, &'static str, Self)> {
        Self::SUITE
            .iter()
            .enumerate()
            .map(|(i, &str)| (i, str, Self::parse(str)))
    }

    /// Test that our [`SUITE`](Suite::SUITE) is well-formatted.
    fn _suite() {
        for j in 1..Self::SUITE.len() {
            let i = j - 1;
            let fst = &Self::SUITE[i];
            let snd = &Self::SUITE[j];
            assert!(
                fst > snd,
                "suite fail at {i}, {j}: {fst} not greater than {snd}",
            );
        }

        for (i, str) in Self::SUITE.iter().enumerate() {
            assert_eq!(
                str,
                &Self::normalize(str),
                "suite fail at {i}: set {str} is not normalized"
            );
        }
    }

    /// Test [`SetTrait::empty`].
    fn _empty() {
        Self::empty().roundtrip("{}");
    }

    /// Test [`SetTrait::singleton`].
    fn _singleton() {
        for (_, str, set) in Self::suite() {
            set.singleton().roundtrip(&format!("{{{str}}}"));
        }
    }

    /// Test [`SetTrait::pair`].
    fn _pair() {
        for (i, str_1, set_1) in Self::suite() {
            for (_, str_2, set_2) in Self::suite().skip(i + 1) {
                set_1
                    .clone()
                    .pair(set_2.clone())
                    .roundtrip(&format!("{{{str_1}, {str_2}}}"));
            }
        }
    }

    /// Test [`PartialEq::eq`].
    fn _eq() {
        for (i, _, set_1) in Self::suite() {
            for (j, _, set_2) in Self::suite() {
                assert_beq!(
                    i == j,
                    set_1 == set_2,
                    "set equality fail at {i}, {j}: {set_1}{}equal to {set_2}"
                );

                assert_beq!(
                    i == j,
                    set_1.partial_cmp(&set_2) == Some(Ordering::Equal),
                    "set equality fail at {i}, {j}: {set_1}{}equal to {set_2}"
                );
            }
        }
    }

    /// Test [`SetTrait::subset`].
    fn _subset() {
        for (i, _, set_1) in Self::suite() {
            for (j, _, set_2) in Self::suite() {
                let subset = set_1.iter().all(|s| set_1.count(s) <= set_2.count(s));

                assert_beq!(
                    subset,
                    set_1 <= set_2,
                    "set equality fail at {i}, {j}: {set_1}{}a subset of {set_2}"
                );

                assert_beq!(
                    subset,
                    set_1.partial_cmp(&set_2).is_some_and(Ordering::is_le),
                    "set equality fail at {i}, {j}: {set_1}{}a subset of {set_2}"
                );
            }
        }
    }

    /// Test [`SetTrait::contains`].
    fn _contains() {
        for (i, _, set_1) in Self::suite() {
            for (j, _, set_2) in Self::suite() {
                let exp = Self::MEM.contains(&(i, j));
                assert_beq!(
                    exp,
                    set_2.contains(&set_1),
                    "set membership fail at {i}, {j}: {set_1}{}a member of {set_2}"
                );
            }
        }
    }

    /// Test [`SetTrait::nat`].
    fn _nat() {
        let mut outputs = Vec::<String>::new();
        for n in 0..5 {
            // Build naturals manually.
            let mut str = String::from('{');
            let mut iter = outputs.iter();
            if let Some(fst) = iter.next() {
                str.push_str(fst);
            }
            for set in iter {
                str.push_str(", ");
                str.push_str(set);
            }
            str.push('}');

            Self::nat(n).roundtrip(&str);
            outputs.push(str);
        }
    }

    /// Test [`SetTrait::sum`].
    fn _sum();

    /// Test [`SetTrait::union`].
    fn _union() {
        for (i, _, set_1) in Self::suite() {
            for (j, _, set_2) in Self::suite() {
                let union = set_1.clone().union(set_2.clone());
                for set in [&set_1, &set_2] {
                    assert!(
                        set.subset(&union),
                        "union fail at {i}, {j}: {set} not a subset of {union}"
                    );
                }
            }
        }
    }

    /// Test [`SetTrait::inter`].
    fn _inter() {
        for (i, _, set_1) in Self::suite() {
            for (j, _, set_2) in Self::suite() {
                let inter = set_1.clone().inter(set_2.clone());
                for set in [&set_1, &set_2] {
                    assert!(
                        inter.subset(set),
                        "intersection fail at {i}, {j}: {inter} not a subset of {set}"
                    );
                }
            }
        }
    }

    /// Test [`SetTrait::powerset`].
    fn _powerset() {
        for (i, _, set) in Self::suite() {
            for subset in Self::powerset(set.clone()) {
                assert!(
                    subset.subset(&set),
                    "powerset fail at {i}: {subset} not a subset of {set}"
                );
            }
        }
    }

    /// Test [`SetTrait::choose`].
    fn _choose() {
        for (i, _, set) in Self::suite() {
            if !set.is_empty() {
                let choose_1 = set.choose().expect("could not choose set");
                assert!(
                    set.contains(choose_1),
                    "choice fail at {i}: {choose_1} not an element of {set}"
                );

                let choose_2 = set.choose_uniq().expect("could not choose set");
                assert!(
                    set.contains(choose_2),
                    "unique choice fail at {i}: {choose_2} not an element of {set}"
                );
            }
        }
    }
}

impl Suite for Mset {
    const SUITE: &'static [&'static str] = &[
        "{}",
        "{{}}",
        "{{}, {}}",
        "{{}, {{}}}",
        "{{}, {{}}, {{}, {{}}}}",
        "{{{}, {}}, {{}, {}}}",
        "{{{{{}}}}}",
    ];

    #[rustfmt::skip]
    const MEM: &'static [(usize, usize)] = &[
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 5), (3, 4)
    ];

    fn _sum() {
        // Remove initial parentheses.
        let suite: Vec<_> = Self::suite()
            .map(|(_, str, set)| (&str[1..(str.len() - 1)], set))
            .collect();

        for (str_1, set_1) in suite.iter() {
            for (str_2, set_2) in suite.iter() {
                set_1
                    .clone()
                    .sum(set_2.clone())
                    .roundtrip(&Self::normalize(&format!("{{{str_1}, {str_2}}}")));
            }
        }
    }
}

impl Suite for Set {
    const SUITE: &'static [&'static str] = &[
        "{}",
        "{{}}",
        "{{}, {{}}}",
        "{{}, {{}}, {{}, {{}}}}",
        "{{}, {{}}, {{{}}}}",
        "{{{}, {{}}}, {{}, {{{}}}}}",
        "{{{{{}}}}}",
    ];

    #[rustfmt::skip]
    const MEM: &'static [(usize, usize)] = &[
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 5)
    ];

    fn _sum() {
        Self::_union();
    }
}

// Run all the suite functions for [`Mset`] and [`Set`].
test!(
    _suite, _empty, _singleton, _pair, _eq, _subset, _contains, _nat, _sum, _union, _inter,
    _powerset, _choose
);

/// Test [`Mset::is_set`].
#[test]
fn mset_is_set() {
    /// Whether each multiset in the suite is a set.
    const IS_SET: &[bool] = &[true, true, false, true, true, false, true];

    for (i, _, set) in Mset::suite() {
        assert_beq!(
            IS_SET[i],
            set.is_set(),
            "is_set fail at {i}: multiset is{}a set"
        );
    }
}

/// Test that each [`Set`] in its suite is a set.
#[test]
fn set_is_set() {
    for (i, _, set) in Set::suite() {
        assert!(
            set.mset().is_set(),
            "is_set fail at {i}: multiset is not a set"
        );
    }
}

/// Test [`Mset::flatten`].
#[test]
fn mset_flatten() {
    /// Flattened suite.
    const FLATTEN: &'static [&'static str] = &[
        "{}",
        "{{}}",
        "{{}}",
        "{{}, {{}}}",
        "{{}, {{}}, {{}, {{}}}}",
        "{{{}}}",
        "{{{{{}}}}}",
    ];

    for (i, _, set) in Mset::suite() {
        set.flatten().roundtrip(FLATTEN[i])
    }
}

/// Test [`Set::kpair`].
#[test]
fn set_kpair() {
    for (i, _, set_1) in Set::suite() {
        for (j, _, set_2) in Set::suite() {
            let pair = Set::kpair(set_1.clone(), set_2.clone());
            assert_eq!(
                pair.ksplit().expect("could not split pair").pair(),
                (&set_1, &set_2),
                "kpair fail at {i}, {j}: pair not split correctly"
            );

            assert_eq!(
                pair.into_ksplit()
                    .expect("could not split pair")
                    .into_pair(),
                (set_1.clone(), set_2),
                "kpair fail at {i}, {j}: pair not split correctly"
            );
        }
    }
}
