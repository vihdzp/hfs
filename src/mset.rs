//! Hereditarily finite multisets [`Mset`].

use crate::prelude::*;
use std::cmp::Ordering;

/// A hereditarily finite multiset.
#[derive(Clone, Default, Eq, IntoIterator)]
pub struct Mset(#[into_iterator(owned, ref, ref_mut)] pub Vec<Mset>);

impl FromIterator<Mset> for Mset {
    fn from_iter<T: IntoIterator<Item = Mset>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl Debug for Mset {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_char('(')?;
        for el in self {
            write!(f, "{el:?}")?;
        }
        f.write_char(')')
    }
}

impl Display for Mset {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.ahu())
    }
}

/// Error in parsing a multiset. This can only happen due to mismatched brackets.
#[derive(Clone, Copy, Debug)]
pub struct Error;

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("mismatched brackets")
    }
}

impl std::error::Error for Error {}

/// Sets are parsed from their set-builder notation. Any symbol other than `{` and `}` is ignored.
impl FromStr for Mset {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Error> {
        let mut stack = Vec::new();
        let mut iter = s.chars();

        loop {
            let c = iter.next().ok_or(Error)?;
            match c {
                // New set.
                '{' => stack.push(Self::empty()),

                // Close last set.
                '}' => {
                    let last = stack.pop().ok_or(Error)?;
                    if let Some(prev) = stack.last_mut() {
                        prev.insert_mut(last);
                    } else {
                        // Set has been built.
                        for c in iter {
                            if ['{', '}'].contains(&c) {
                                return Err(Error);
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

/// Two multisets are equal if they contain the same elements the same number of times.
impl PartialEq for Mset {
    fn eq(&self, other: &Self) -> bool {
        if let Some((fst, snd)) =
            Levels::init(self).both(Levels::init(other), |fst, snd| fst.len() == snd.len())
        {
            // Since both sets have the same number of hereditary subsets, checking the subset
            // relation suffices.
            fst.subset(&snd)
        } else {
            false
        }
    }
}

/// A multiset is smaller or equal than another when it contains the same elements at most as many
/// times as the other.
impl PartialOrd for Mset {
    fn le(&self, other: &Self) -> bool {
        let levels =
            Levels::init(self).both(Levels::init(other), |fst, snd| fst.len() <= snd.len());

        if let Some((fst, snd)) = levels {
            fst.subset(&snd)
        } else {
            false
        }
    }

    fn ge(&self, other: &Self) -> bool {
        other.le(self)
    }

    fn lt(&self, other: &Self) -> bool {
        let levels =
            Levels::init(self).both(Levels::init(other), |fst, snd| fst.len() <= snd.len());

        if let Some((fst, snd)) = levels {
            fst.len() < snd.len() && fst.subset(&snd)
        } else {
            false
        }
    }

    fn gt(&self, other: &Self) -> bool {
        other.lt(self)
    }

    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut candidate = Ordering::Equal;
        let levels = Levels::init(self).both(Levels::init(other), |fst, snd| {
            let cmp = fst.len().cmp(&snd.len());
            if cmp.is_ne() {
                if candidate.is_eq() {
                    candidate = cmp;
                } else if candidate != cmp {
                    return false;
                }
            }

            true
        });

        if let Some((fst, snd)) = levels {
            // If the code reaches this point, the candidate ordering is the only possible ordering.
            let test = match candidate {
                Ordering::Less | Ordering::Equal => fst.subset(&snd),
                Ordering::Greater => snd.subset(&fst),
            };

            if test {
                return Some(candidate);
            }
        }

        None
    }
}

impl Mset {
    /// The empty set Ø.
    #[must_use]
    pub const fn empty() -> Self {
        Self(Vec::new())
    }

    /// Returns whether the multiset is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Clears the multiset.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Set cardinality.
    #[must_use]
    pub fn card(&self) -> usize {
        self.0.len()
    }

    /// An iterator over the elements of the [`Mset`].
    pub fn iter(&self) -> std::slice::Iter<Mset> {
        self.into_iter()
    }

    /// A mutable iterator over the elements of the [`Mset`].
    pub fn iter_mut(&mut self) -> std::slice::IterMut<Mset> {
        self.into_iter()
    }

    /// Finds the [`Ahu`] encoding for a multiset.
    #[must_use]
    pub fn ahu(&self) -> Ahu {
        Ahu::new(self)
    }

    /// Set membership ∈.
    #[must_use]
    pub fn mem(&self, other: &Self) -> bool {
        let mut fst = unsafe { Levels::empty() };
        let snd = Levels::init(other).new();
        let mut buf = Vec::new();

        self.iter().any(move |set| {
            fst.init_mut(set);
            let mut r = 1;
            while fst.step(&mut buf) {
                if let Some(level) = snd.get(r) {
                    if fst.last().len() != level.len() {
                        return false;
                    }
                } else {
                    return false;
                }

                r += 1;
            }

            fst.subset(&snd)
        })
    }

    /// Subset ⊆.
    #[must_use]
    pub fn subset(&self, other: &Self) -> bool {
        self.le(other)
    }

    /// Mutable set insertion.
    pub fn insert_mut(&mut self, other: Self) {
        self.0.push(other);
    }

    /// Set insertion x ∪ {y}.
    #[must_use]
    pub fn insert(mut self, other: Self) -> Self {
        self.insert_mut(other);
        self
    }

    /// Set singleton {x}.
    #[must_use]
    pub fn singleton(self) -> Self {
        Self(vec![self])
    }

    /// Set pair {x, y}.
    #[must_use]
    pub fn pair(self, other: Self) -> Self {
        Self(vec![self, other])
    }

    /// Set union x ∪ y.
    #[must_use]
    pub fn union(mut self, other: Self) -> Self {
        self.0.extend(other);
        self
    }

    /// Set union ∪x.
    #[must_use]
    pub fn big_union(self) -> Self {
        self.into_iter().flatten().collect()
    }

    /// Mutable set specification.
    pub fn select_mut<P: FnMut(&Mset) -> bool>(&mut self, mut pred: P) {
        let mut i = 0;
        while i < self.card() {
            if pred(&self.0[i]) {
                i += 1;
            } else {
                self.0.swap_remove(i);
            }
        }
    }

    /// Set specification.
    #[must_use]
    pub fn select<P: FnMut(&Mset) -> bool>(mut self, pred: P) -> Self {
        self.select_mut(pred);
        self
    }

    /// Set intersection x ∩ y.
    #[must_use]
    pub fn inter(self, other: Self) -> Self {
        let idx = self.card();
        let mut pair = self.pair(other);
        let levels = Levels::init(&pair).new();

        // The intersection of two empty sets is empty.
        let elements;
        if let Some(els) = levels.get(2) {
            elements = els;
        } else {
            return Self::empty();
        }

        // We store the indices of the sets in the intersection.
        let (mut next, mut indices) = levels.mod_ahu(3);

        let mut sets: BTreeMap<_, SmallVec<_>> = BTreeMap::new();
        for (i, range) in Levels::child_iter(elements).enumerate() {
            let slice = unsafe {
                let slice = next.get_unchecked_mut(range);
                slice.sort_unstable();
                slice as &[_]
            };

            // Each entry stores the indices where it's found within the first set.
            let children: SmallVec<_> = slice.iter().copied().collect();
            match sets.entry(children) {
                Entry::Vacant(entry) => {
                    if i < idx {
                        entry.insert(smallvec![i]);
                    }
                }
                Entry::Occupied(mut entry) => {
                    if i < idx {
                        entry.get_mut().push(i);
                    } else if let Some(j) = entry.get_mut().pop() {
                        indices.push(j);
                    }
                }
            }
        }

        let mut snd = unsafe { pair.0.pop().unwrap_unchecked() };
        let mut fst = unsafe { pair.0.pop().unwrap_unchecked() };
        snd.clear();

        for i in indices {
            let set = std::mem::take(unsafe { fst.0.get_unchecked_mut(i) });
            snd.insert_mut(set);
        }

        snd
    }

    /// Powerset 2^x.
    #[must_use]
    pub fn powerset(self) -> Self {
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

    /// The von Neumann rank of the set.
    #[must_use]
    pub fn rank(&self) -> usize {
        Levels::init(self).new().rank()
    }

    /// The von Neumann set encoding for n.
    #[must_use]
    pub fn nat(n: usize) -> Self {
        let mut res = Mset::empty();
        for _ in 0..n {
            res.insert_mut(res.clone());
        }
        res
    }

    /// The Zermelo set encoding for n.
    #[must_use]
    pub fn zermelo(n: usize) -> Self {
        let mut res = Mset::empty();
        for _ in 0..n {
            res = res.singleton();
        }
        res
    }

    /// The von Neumann hierarchy.
    #[must_use]
    pub fn neumann(n: usize) -> Self {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    const NATS: [&str; 4] = ["{}", "{{}}", "{{}, {{}}}", "{{}, {{}}, {{}, {{}}}}"];

    /// Verify round-trip between set and string.
    fn roundtrip(set: Mset, str: &str) {
        assert_eq!(set.to_string(), str);
        assert_eq!(set, str.parse().unwrap());
    }

    #[test]
    fn empty() {
        roundtrip(Mset::empty(), "{}");
    }

    #[test]
    fn singleton() {
        roundtrip(Mset::empty().singleton(), "{{}}");
    }

    #[test]
    fn pair() {
        let set = Mset::empty();
        roundtrip(set.clone().pair(set), "{{}, {}}");
    }

    #[test]
    fn nat() {
        for n in 0..4 {
            roundtrip(Mset::nat(n), NATS[n]);
        }
    }

    #[test]
    fn union() {
        let set = Mset::nat(2).union(Mset::nat(3));
        roundtrip(set, "{{}, {}, {{}}, {{}}, {{}, {{}}}}");
    }
}
