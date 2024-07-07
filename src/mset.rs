//! Hereditarily finite multisets [`Mset`].

use crate::prelude::*;

/// A [hereditarily finite](https://en.wikipedia.org/wiki/Hereditarily_finite_set)
/// [multiset](https://en.wikipedia.org/wiki/Multiset).
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

    unsafe fn _as_slice_mut(&mut self) -> &mut [Self] {
        &mut self.0
    }

    fn as_vec(&self) -> &Vec<Mset> {
        &self.0
    }

    unsafe fn _as_vec_mut(&mut self) -> &mut Vec<Mset> {
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

    fn sum_vec(vec: Vec<Self>) -> Self {
        Self::sum_iter(vec)
    }

    fn sum(mut self, mut other: Self) -> Self {
        self.0.append(&mut other.0);
        self
    }

    fn inter_vec(mut vec: Vec<Self>) -> Option<Self> {
        // Check for trivial cases.
        match vec.len() {
            0 => return None,
            1 => return Some(unsafe { vec.pop().unwrap_unchecked() }),
            _ => {}
        }
        let levels =
            unsafe { Levels::init_iter(vec.iter().map(AsRef::as_ref)).unwrap_unchecked() }.fill();

        let mut next = levels.ahu(1);
        let mut iter = unsafe { Levels::child_iter(levels.first(), &mut next) };

        // Each entry stores the indices where it's found within the first set, and a counter for
        // how many times it's been seen in every other set.
        let mut sets = BTreeMap::new();
        let fst = unsafe { iter.next().unwrap_unchecked() };
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
            for set in slice.iter() {
                match sets.entry(*set) {
                    Entry::Vacant(_) => {}
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().1 += 1;
                    }
                }
            }

            // Update counts.
            for (indices, count) in sets.values_mut() {
                indices.truncate(*count);
                *count = 0;
            }
        }

        // Take elements from the first set, reuse some other set as a buffer.
        let mut fst = vec.swap_remove(0);
        let mut snd = vec.swap_remove(1);
        snd.clear();

        for (indices, _) in sets.into_values() {
            for i in indices {
                let set = std::mem::take(unsafe { fst._as_slice_mut().get_unchecked_mut(i) });
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
    pub fn as_slice_mut(&mut self) -> &mut [Self] {
        &mut self.0
    }

    /// A mutable reference to the inner vector.
    pub fn as_vec_mut(&mut self) -> &mut Vec<Self> {
        &mut self.0
    }

    /// Mutably iterate over the elements of the set.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<Self> {
        self.0.iter_mut()
    }

    /// Sum over an iterator.
    pub fn sum_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter().flatten().collect()
    }
}

/// Tests for [`Mset`].
#[cfg(test)]
mod mset {
    use super::*;

    #[test]
    fn basic() {
        println!("{}", Mset::nat(3).pair(Mset::nat(3)));
    }

    /// A multitude of general multisets for general-purpose testing.
    ///
    /// Both the list and each constituent set should be normalized, i.e. in decreasing
    /// lexicographic order.
    const SUITE: &[&str] = &[
        "{}",
        "{{}}",
        "{{}, {}}",
        "{{}, {{}}, {{}, {{}}}}",
        "{{{}, {}}, {{}, {}}}",
        "{{{{{}}}}}",
    ];

    /// Our [`SUITE`] as `(&str, Mset)` pairs.
    fn suite() -> impl Iterator<Item = (&'static str, Mset)> {
        SUITE.iter().map(|&str| (str, str.parse().unwrap()))
    }

    /// Normalize string for a set.
    fn normalize(str: &str) -> String {
        let set: Mset = str.parse().unwrap();
        set.to_string()
    }

    fn inner(str: &str) -> &str {
        &str[1..(str.len() - 1)]
    }

    /// Verify round-trip conversion between set and string.
    fn roundtrip(set: &Mset, str: &str) {
        assert_eq!(set, &str.parse().unwrap());
        assert_eq!(set.to_string(), str);
    }

    /// Normalize string, then call [`roundtrip`].
    fn roundtrip_norm(set: &Mset, str: &str) {
        roundtrip(set, &normalize(str))
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

        let _ = suite();
    }

    /// Test [`Mset::empty`].
    #[test]
    fn empty() {
        roundtrip(&Mset::empty(), "{}");
    }

    /// Test [`Mset::singleton`].
    #[test]
    fn singleton() {
        for (str, set) in suite() {
            roundtrip(&set.singleton(), &format!("{{{str}}}"));
        }
    }

    /// Test [`Mset::pair`].
    #[test]
    fn pair() {
        for (i, (str_1, set_1)) in suite().enumerate() {
            for (str_2, set_2) in suite().skip(i) {
                roundtrip(
                    &set_1.clone().pair(set_2.clone()),
                    &format!("{{{str_1}, {str_2}}}"),
                );
            }
        }
    }

    /// Test [`Mset::mem`].
    #[test]
    fn mem() {
        const MEM: &[(usize, usize)] = &[(0, 1), (0, 2), (0, 3), (1, 3), (2, 4)];

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

            roundtrip(&Mset::nat(n), &str);
            outputs.push(str);
        }
    }

    /// Test [`Mset::union`].
    #[test]
    fn union() {
        // Remove initial parentheses.
        let suite = || suite().map(|(str, set)| (inner(str), set));

        for (str_1, set_1) in suite() {
            for (str_2, set_2) in suite() {
                roundtrip_norm(
                    &set_1.clone().union(set_2.clone()),
                    &format!("{{{str_1}, {str_2}}}"),
                );
            }
        }
    }

    /// Test [`Mset::inter`].
    #[test]
    fn inter() {
        for (_, set_1) in suite() {
            for (_, set_2) in suite() {
                let inter = set_1.clone().inter(set_2.clone());
                for set in inter {
                    assert!(set.contains(&set_1));
                    assert!(set.contains(&set_2))
                }
            }
        }
    }
}
