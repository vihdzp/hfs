//! Hereditarily finite multisets [`Mset`].

use std::cmp::Ordering;

use crate::{prelude::*, utils::Levels};

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
pub struct MsetError;

impl Display for MsetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("mismatched brackets")
    }
}

impl Error for MsetError {}

/// Sets are parsed from their set-builder notation. Any symbol other than `{` and `}` is ignored.
impl FromStr for Mset {
    type Err = MsetError;

    fn from_str(s: &str) -> Result<Self, MsetError> {
        let mut stack = Vec::new();
        let mut iter = s.chars();

        loop {
            let c = iter.next().ok_or(MsetError)?;
            match c {
                // New set.
                '{' => stack.push(Self::empty()),

                // Close last set.
                '}' => {
                    let last = stack.pop().ok_or(MsetError)?;
                    if let Some(prev) = stack.last_mut() {
                        prev.insert_mut(last);
                    } else {
                        // Set has been built.
                        while let Some(c) = iter.next() {
                            if ['{', '}'].contains(&c) {
                                return Err(MsetError);
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
            Levels::both(self, other, Vec::extend, |fst, snd| fst.len() == snd.len())
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
        if let Some((fst, snd)) =
            Levels::both(self, other, Vec::extend, |fst, snd| fst.len() <= snd.len())
        {
            fst.subset(&snd)
        } else {
            false
        }
    }

    fn ge(&self, other: &Self) -> bool {
        other.le(self)
    }

    fn lt(&self, other: &Self) -> bool {
        if let Some((fst, snd)) =
            Levels::both(self, other, Vec::extend, |fst, snd| fst.len() <= snd.len())
        {
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
        let levels = Levels::both(self, other, Vec::extend, |fst, snd| {
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
    pub const fn empty() -> Self {
        Self(Vec::new())
    }

    /// Returns whether the multiset is finite.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Set cardinality.
    pub fn card(&self) -> usize {
        self.0.len()
    }

    /// An iterator over the elements of the [`Mset`].
    pub fn iter(&self) -> std::slice::Iter<Mset> {
        self.0.iter()
    }

    /// A mutable iterator over the elements of the [`Mset`].
    pub fn iter_mut(&mut self) -> std::slice::IterMut<Mset> {
        self.0.iter_mut()
    }

    /// Finds the [`Ahu`] encoding for a multiset.
    pub fn ahu(&self) -> Ahu {
        Ahu::new(self)
    }

    /// Set membership ∈.
    pub fn mem(&self, other: &Self) -> bool {
        self.iter().find(|&set| set == other).is_some()
    }

    /// Subset ⊆.
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
    pub fn singleton(self) -> Self {
        Self(vec![self])
    }

    /// Set pair {x, y}.
    pub fn pair(self, other: Self) -> Self {
        Self(vec![self, other])
    }

    /// Set union x ∪ y.
    pub fn union(mut self, other: Self) -> Self {
        self.0.extend(other);
        self
    }

    /// Set union ∪x.
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
    pub fn select<P: FnMut(&Mset) -> bool>(mut self, pred: P) -> Self {
        self.select_mut(pred);
        self
    }

    // Set intersection x ∩ y.
    pub fn intersection(self, other: Self) -> Self {
        todo!()
    }

    /// Powerset 2^x.
    pub fn powerset(self) -> Self {
        let n = self.card();
        let mut powerset = Self(vec![Self::empty()]);
        if n == 0 {
            return powerset;
        }

        for mut i in 1..((1 << n) - 1) {
            let mut subset = Mset::empty();
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
    pub fn rank(&self) -> usize {
        Levels::new(self).rank()
    }

    /// The von Neumann set encoding for n.
    pub fn nat(n: usize) -> Self {
        let mut res = Mset::empty();
        for _ in 0..n {
            res.insert_mut(res.clone());
        }
        res
    }

    /// The Zermelo set encoding for n.
    pub fn zermelo(n: usize) -> Self {
        let mut res = Mset::empty();
        for _ in 0..n {
            res = res.singleton();
        }
        res
    }

    /// The von Neumann hierarchy.
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
