//! Hereditarily finite multisets [`Mset`].

use crate::prelude::*;

/// A [hereditarily finite](https://en.wikipedia.org/wiki/Hereditarily_finite_set)
/// [multiset](https://en.wikipedia.org/wiki/Multiset).
///
/// Unlike sets, multisets can contain an element multiple times. The number of times they do is
/// referred to as their multiplicity or [count](Mset::count). Familiar operations on sets, like
/// unions and intersections, are reinterpreted in terms of these multiplicities.
///
/// ## Internal representation
///
/// Each [`Mset`] contains only a `Vec` of [`Mset`]. Rust's ownership system guarantees that [quine
/// atoms](https://en.wikipedia.org/wiki/Urelement#Quine_atoms), or any other non-regular multisets
/// cannot exist. In other words, a multiset can't contain itself.
#[derive(Clone, Default, Eq, IntoIterator)]
pub struct Mset(#[into_iterator(owned, ref, ref_mut)] pub Vec<Mset>);

// -------------------- Basic traits -------------------- //

impl AsRef<Mset> for Mset {
    fn as_ref(&self) -> &Mset {
        self
    }
}

impl From<Mset> for Vec<Mset> {
    fn from(set: Mset) -> Self {
        set.0
    }
}

impl FromIterator<Mset> for Mset {
    fn from_iter<T: IntoIterator<Item = Mset>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// Succintly writes a multiset as stored in memory.
impl Debug for Mset {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_char('(')?;
        for el in self {
            write!(f, "{el:?}")?;
        }
        f.write_char(')')
    }
}

/// Displays a multiset in canonical roster notation.
impl Display for Mset {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.ahu())
    }
}

// -------------------- String parsing -------------------- //

/// Error in parsing a set. This can only happen due to mismatched brackets.
#[derive(Clone, Copy, Debug)]
pub struct SetError;

impl Display for SetError {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.write_str("mismatched brackets")
    }
}

impl std::error::Error for SetError {}

/// Multisets are parsed from their roster notation. Any symbol other than `{` and `}` is ignored,
/// including commas.
impl FromStr for Mset {
    type Err = SetError;

    fn from_str(s: &str) -> Result<Self, SetError> {
        let mut stack = Vec::new();
        let mut iter = s.chars();

        loop {
            let c = iter.next().ok_or(SetError)?;
            match c {
                // New multiset.
                '{' => stack.push(Self::empty()),

                // Close last multiset.
                '}' => {
                    let last = stack.pop().ok_or(SetError)?;
                    if let Some(prev) = stack.last_mut() {
                        prev.insert_mut(last);
                    } else {
                        // Multiset has been built.
                        for c in iter {
                            if ['{', '}'].contains(&c) {
                                return Err(SetError);
                            }
                        }

                        return Ok(last);
                    }
                }
                _ => {}
            }
        }
    }
}

// -------------------- SetTrait -------------------- //

impl crate::Seal for Mset {}

impl SetTrait for Mset {
    // -------------------- Basic methods -------------------- //

    fn as_slice(&self) -> &[Self] {
        &self.0
    }

    unsafe fn _as_mut_slice(&mut self) -> &mut [Self] {
        &mut self.0
    }

    fn as_vec(&self) -> &Vec<Mset> {
        &self.0
    }

    unsafe fn _as_mut_vec(&mut self) -> &mut Vec<Mset> {
        &mut self.0
    }

    // -------------------- Constructions -------------------- //

    fn empty() -> Self {
        Self(Vec::new())
    }

    fn singleton(self) -> Self {
        Self(vec![self])
    }

    fn insert_mut(&mut self, other: Self) {
        self.0.push(other);
    }

    fn select_mut<P: FnMut(&Mset) -> bool>(&mut self, mut pred: P) {
        let mut i = 0;
        while i < self.card() {
            if pred(&self.0[i]) {
                i += 1;
            } else {
                self.0.swap_remove(i);
            }
        }
    }

    fn sum(mut self, mut other: Self) -> Self {
        self.0.append(&mut other.0);
        self
    }

    fn sum_vec(vec: Vec<Self>) -> Self {
        Self::sum_iter(vec)
    }

    fn union_vec(mut vec: Vec<Self>) -> Self {
        // Check for trivial cases.
        match vec.len() {
            0 => return Self::empty(),
            1 => return vec.pop().unwrap(),
            _ => {}
        }

        let levels = Levels::init_iter(&vec).unwrap().fill();
        let next = levels.ahu(1);
        // Safety: the length of `next` is exactly the sum of cardinalities in the first level.
        let mut iter = unsafe { Levels::child_iter(levels.first(), &next) }.enumerate();

        // Each entry stores indices to where it's found in `vec`, and a counter for how many times
        // it's been seen in every other set.
        //
        // It should be possible to use a single pointer instead of two indices. But I've got no
        // idea how to make this work safely.
        //
        // Safety: we already know there's at least 2 sets.
        let fst = unsafe { iter.next().unwrap_unchecked().1 };
        let mut sets = BTreeMap::new();
        for (i, set) in fst.iter().enumerate() {
            let el_idx = (0, i);
            match sets.entry(*set) {
                Entry::Vacant(entry) => {
                    entry.insert((smallvec![el_idx], 0));
                }
                Entry::Occupied(mut entry) => {
                    entry.get_mut().0.push(el_idx);
                }
            }
        }

        // Count number of appearances in other sets.
        for (n, slice) in iter {
            for (i, set) in slice.iter().enumerate() {
                let el_idx = (n, i);
                match sets.entry(*set) {
                    Entry::Vacant(entry) => {
                        entry.insert((smallvec![el_idx], 1));
                    }
                    Entry::Occupied(mut entry) => {
                        let (indices, count) = entry.get_mut();
                        if indices.len() == *count {
                            indices.push(el_idx);
                        }
                        *count += 1;
                    }
                }
            }

            // Reset counts.
            for (_, count) in sets.values_mut() {
                *count = 0;
            }
        }

        // Get all the sets we need.
        let mut union = Self::empty();
        for (indices, _) in sets.into_values() {
            for (n, i) in indices {
                // Safety: all the indices we built are valid for this sort of indexing.
                let set = mem::take(unsafe { vec.get_unchecked_mut(n).0.get_unchecked_mut(i) });
                union.insert_mut(set);
            }
        }

        union
    }

    fn inter_vec(mut vec: Vec<Self>) -> Option<Self> {
        // Check for trivial cases.
        match vec.len() {
            0 => return None,
            1 => return Some(vec.pop().unwrap()),
            _ => {}
        }
        let levels = Levels::init_iter(&vec).unwrap().fill();

        let next = levels.ahu(1);
        // Safety: the length of `next` is exactly the sum of cardinalities in the first level.
        let mut iter = unsafe { Levels::child_iter(levels.first(), &next) };

        // Each entry stores the indices where it's found within the first set, and a counter for
        // how many times it's been seen in every other set.
        //
        // Safety: we already know there's at least 2 sets.
        let fst = unsafe { iter.next().unwrap_unchecked() };
        let mut sets = BTreeMap::new();
        for (i, set) in fst.iter().enumerate() {
            match sets.entry(*set) {
                Entry::Vacant(entry) => {
                    entry.insert((smallvec![i], 0));
                }
                Entry::Occupied(mut entry) => {
                    entry.get_mut().0.push(i);
                }
            }
        }

        // Count number of appearances in other sets.
        for slice in iter {
            for set in slice {
                match sets.entry(*set) {
                    Entry::Vacant(_) => {}
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().1 += 1;
                    }
                }
            }

            // Update counts.
            sets.retain(|_, (indices, count)| {
                indices.truncate(*count);
                let retain = *count != 0;
                *count = 0;
                retain
            });
        }

        // Take elements from the first set, reuse some other set as a buffer.
        let mut fst = vec.swap_remove(0);
        let mut snd = vec.swap_remove(0);
        snd.clear();

        for (indices, _) in sets.into_values() {
            for i in indices {
                // Safety: all the indices we built are valid for the first set.
                let set = mem::take(unsafe { fst._as_mut_slice().get_unchecked_mut(i) });
                snd.insert_mut(set);
            }
        }

        Some(snd)
    }

    fn powerset(self) -> Self {
        let n = self.card();
        let mut powerset = Self::empty().singleton();
        if n == 0 {
            return powerset;
        }

        for mut i in 1..((1 << n) - 1) {
            let mut subset = Self::empty();
            for j in 0..n {
                if i % 2 == 1 {
                    subset.insert_mut(self.0[j].clone());
                }
                i /= 2;
            }

            powerset.insert_mut(subset);
        }

        powerset.insert(self)
    }

    fn nat(n: usize) -> Self {
        let mut res = Mset::empty();
        for _ in 0..n {
            res.insert_mut(res.clone());
        }
        res
    }

    fn zermelo(n: usize) -> Self {
        let mut res = Mset::empty();
        for _ in 0..n {
            res = res.singleton();
        }
        res
    }

    fn neumann(n: usize) -> Self {
        debug_assert!(
            n <= 5,
            "the sixth set in the von Neumann hierarchy has 2^65536 elements"
        );

        let mut set = Self::empty();
        for _ in 0..n {
            set = set.powerset();
        }
        set
    }

    // -------------------- Relations -------------------- //

    unsafe fn _levels_subset(fst: &Levels<&Mset>, snd: &Levels<&Mset>) -> bool {
        fst.both_ahu(
            snd,
            // Decrement set count. Return if this reaches a negative.
            |sets, children| match sets.entry(children) {
                Entry::Vacant(_) => None,
                Entry::Occupied(mut entry) => {
                    let (idx, num) = entry.get_mut();
                    let idx = *idx;
                    if *num == 0 {
                        entry.remove_entry();
                    } else {
                        *num -= 1;
                    }
                    Some(idx)
                }
            },
            // Increment set count.
            |sets, children| {
                let len = sets.len();
                match sets.entry(children) {
                    Entry::Vacant(entry) => {
                        entry.insert((len, 0));
                        len
                    }
                    Entry::Occupied(mut entry) => {
                        let (idx, num) = entry.get_mut();
                        *num += 1;
                        *idx
                    }
                }
            },
        )
    }

    fn disjoint_iter<'a, I: IntoIterator<Item = &'a Self>>(_iter: I) -> bool {
        todo!()
    }

    fn disjoint_pairwise<'a, I: IntoIterator<Item = &'a Self>>(_iter: I) -> bool {
        todo!()
    }
}

// -------------------- Other -------------------- //

impl Mset {
    /// The set as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [Self] {
        &mut self.0
    }

    /// A mutable reference to the inner vector.
    pub fn as_mut_vec(&mut self) -> &mut Vec<Self> {
        &mut self.0
    }

    /// Mutably iterate over the elements of the set.
    pub fn iter_mut(&mut self) -> slice::IterMut<Self> {
        self.0.iter_mut()
    }

    /// Sum over an iterator.
    pub fn sum_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter().flatten().collect()
    }

    /// Count multiplicity of an element in a set.
    #[must_use]
    pub fn count(&self, other: &Self) -> usize {
        self.filter_eq(other).count()
    }
}

/// Tests for [`Mset`].
#[cfg(test)]
mod mset {
    use super::*;

    /// A multitude of general multisets for general-purpose testing.
    ///
    /// Both the list and each constituent set should be normalized, i.e. in decreasing
    /// lexicographic order.
    const SUITE: &[&str] = &[
        "{}",
        "{{}}",
        "{{}, {}}",
        "{{}, {{}}}",
        "{{}, {{}}, {{}, {{}}}}",
        "{{{}, {}}, {{}, {}}}",
        "{{{{{}}}}}",
    ];

    /// Our [`SUITE`] as `(&str, Mset)` pairs.
    fn suite() -> impl Iterator<Item = (&'static str, Mset)> {
        SUITE.iter().map(|&str| (str, str.parse().unwrap()))
    }

    /// Test that our [`SUITE`] is well-formatted.
    #[test]
    fn test_suite() {
        for i in 1..SUITE.len() {
            assert!(
                SUITE[i - 1] > SUITE[i],
                "test suite must be inversely lexicographically ordered"
            )
        }

        for str in SUITE {
            assert_eq!(str, &Mset::_normalize(str), "test suite must round-trip");
        }
    }

    /// Test [`Mset::empty`].
    #[test]
    fn empty() {
        Mset::empty()._roundtrip("{}");
    }

    /// Test [`Mset::singleton`].
    #[test]
    fn singleton() {
        for (str, set) in suite() {
            set.singleton()._roundtrip(&format!("{{{str}}}"));
        }
    }

    /// Test [`Mset::pair`].
    #[test]
    fn pair() {
        for (i, (str_1, set_1)) in suite().enumerate() {
            for (str_2, set_2) in suite().skip(i) {
                set_1
                    .clone()
                    .pair(set_2.clone())
                    ._roundtrip(&format!("{{{str_1}, {str_2}}}"));
            }
        }
    }

    /// Test [`Mset::eq`].
    #[test]
    fn eq() {
        for (i, (_, set_1)) in suite().enumerate() {
            for (j, (_, set_2)) in suite().enumerate() {
                assert_eq!(
                    i == j,
                    set_1 == set_2,
                    "set equality fail: {set_1} | {set_2}"
                )
            }
        }
    }

    /// Test [`Mset::contains`].
    #[test]
    fn contains() {
        /// Hardcoded array of pairs that belong in each other.
        #[rustfmt::skip]
        const MEM: &[(usize, usize)] = &[
            (0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 5), (3, 4)
        ];

        for (i, (_, set_1)) in suite().enumerate() {
            for (j, (_, set_2)) in suite().enumerate() {
                assert_eq!(
                    MEM.contains(&(i, j)),
                    set_2.contains(&set_1),
                    "set membership fail {i}, {j}: {set_1} | {set_2}"
                )
            }
        }
    }

    /// Test [`Mset::nat`].
    #[test]
    fn nat() {
        let mut outputs = Vec::<String>::new();
        for n in 0..5 {
            // Build naturals manually.
            let mut str = String::from('{');
            let mut iter = outputs.iter();
            if let Some(fst) = iter.next() {
                str.push_str(&fst);
            }
            for set in iter {
                str.push_str(", ");
                str.push_str(set);
            }
            str.push('}');

            Mset::nat(n)._roundtrip(&str);
            outputs.push(str);
        }
    }

    /// Test [`Mset::sum`].
    #[test]
    fn sum() {
        // Remove initial parentheses.
        let suite = || suite().map(|(str, set)| (&str[1..(str.len() - 1)], set));

        for (str_1, set_1) in suite() {
            for (str_2, set_2) in suite() {
                set_1
                    .clone()
                    .sum(set_2.clone())
                    ._roundtrip(&Mset::_normalize(&format!("{{{str_1}, {str_2}}}")));
            }
        }
    }

    /// Test [`Mset::union`].
    #[test]
    fn union() {
        for (_, set_1) in suite() {
            for (_, set_2) in suite() {
                let union = set_1.clone().union(set_2.clone());
                for set in [&set_1, &set_2] {
                    assert!(set.subset(&union), "{set} not a subset of {union}");
                }
            }
        }
    }

    /// Test [`Mset::inter`].
    #[test]
    fn inter() {
        for (_, set_1) in suite() {
            for (_, set_2) in suite() {
                let inter = set_1.clone().inter(set_2.clone());
                for set in [&set_1, &set_2] {
                    assert!(inter.subset(set), "{inter} not a subset of {set}");
                }
            }
        }
    }
}
