//! General library tests.

#![cfg(test)]

use crate::prelude::*;
use concat_idents::concat_idents;

/// Creates analogous tests for [`Set`] and [`Mset`].
macro_rules! test {
    ($($name: ident),*) => {
        $(
            concat_idents!(fn_name = mset, $name {
                #[test]
                fn fn_name() {
                    Mset::$name();
                }
            });

            concat_idents!(fn_name = set, $name {
                #[test]
                fn fn_name() {
                    Set::$name();
                }
            });
        )*
    };
}

trait Suite: SetTrait {
    /// A multitude of general sets for general-purpose testing.
    ///
    /// Both the list and each constituent set should be normalized, i.e. in decreasing
    /// lexicographic order.
    const SUITE: &'static [&'static str];

    /// Hardcoded indices of pairs of sets containing each other.
    const MEM: &'static [(usize, usize)];

    /// Normalize string for a set.
    #[must_use]
    fn normalize(str: &str) -> String {
        let set: Self = str.parse().unwrap();
        set.to_string()
    }

    /// Verify round-trip conversion between a set and a string.
    fn roundtrip(&self, str: &str) {
        assert_eq!(self, &str.parse().unwrap());
        assert_eq!(self.to_string(), str);
    }

    /// Our [`SUITE`](Suite::SUITE) as `(&str, Self)` pairs.
    fn suite() -> impl Iterator<Item = (&'static str, Self)> {
        Self::SUITE.iter().map(|&str| (str, str.parse().unwrap()))
    }

    /// Test that our [`SUITE`](Suite::SUITE) is well-formatted.
    fn _suite() {
        for i in 1..Self::SUITE.len() {
            assert!(
                Self::SUITE[i - 1] > Self::SUITE[i],
                "test suite must be inversely lexicographically ordered\n\
                indices {} and {i}",
                i - 1
            );
        }

        for str in Self::SUITE {
            assert_eq!(str, &Self::normalize(str), "test suite must round-trip");
        }
    }

    /// Test [`SetTrait::empty`].
    fn _empty() {
        Self::empty().roundtrip("{}");
    }

    /// Test [`SetTrait::singleton`].
    fn _singleton() {
        for (str, set) in Self::suite() {
            set.singleton().roundtrip(&format!("{{{str}}}"));
        }
    }

    /// Test [`SetTrait::pair`].
    fn _pair() {
        for (i, (str_1, set_1)) in Self::suite().enumerate() {
            for (str_2, set_2) in Self::suite().skip(i + 1) {
                set_1
                    .clone()
                    .pair(set_2.clone())
                    .roundtrip(&format!("{{{str_1}, {str_2}}}"));
            }
        }
    }

    /// Test [`SetTrait::eq`].
    fn _eq() {
        for (i, (_, set_1)) in Self::suite().enumerate() {
            for (j, (_, set_2)) in Self::suite().enumerate() {
                assert_eq!(
                    i == j,
                    set_1 == set_2,
                    "set equality fail at {i}, {j}: {set_1} | {set_2}"
                );
            }
        }
    }

    /// Test [`SetTrait::contains`].
    fn _contains() {
        for (i, (_, set_1)) in Self::suite().enumerate() {
            for (j, (_, set_2)) in Self::suite().enumerate() {
                assert_eq!(
                    Self::MEM.contains(&(i, j)),
                    set_2.contains(&set_1),
                    "set membership fail at {i}, {j}: {set_1} | {set_2}"
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
        for (i, (_, set_1)) in Self::suite().enumerate() {
            for (j, (_, set_2)) in Self::suite().enumerate() {
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

    /// Test [`Mset::inter`].
    fn _inter() {
        for (i, (_, set_1)) in Self::suite().enumerate() {
            for (j, (_, set_2)) in Self::suite().enumerate() {
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
        let suite = || Self::suite().map(|(str, set)| (&str[1..(str.len() - 1)], set));

        for (str_1, set_1) in suite() {
            for (str_2, set_2) in suite() {
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

test!(_suite, _empty, _singleton, _pair, _eq, _contains, _nat, _sum, _union, _inter);
