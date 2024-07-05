//! Hereditarily finite sets [`Set`].

use crate::prelude::*;

/// A set is a multiset that hereditarily has no duplicate elements.
///
/// ## Invariants
///
/// Every two elements in a [`Set`] must be distinct. Moreover, every element must satisfy this same
/// guarantee.
#[derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Set(Mset);

impl AsRef<Mset> for Set {
    fn as_ref(&self) -> &Mset {
        &self.0
    }
}

impl From<Set> for Mset {
    fn from(value: Set) -> Self {
        value.0
    }
}

impl Debug for Set {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{:?}", self.mset())
    }
}

impl Display for Set {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.mset())
    }
}

// impl iterator.

/// A hybrid set, i.e. a multiset of sets.
//struct Hset(Vec<Set>);

/// Orders and deduplicates a set based on the corresponding keys.
///
/// ## Safety
///
/// Both `set` and `keys` must have the same number of elements.
fn dedup_by<T: Default>(
    set: &mut Vec<T>,
    keys: &[usize],
    buf1: &mut Vec<(usize, usize)>,
    buf2: &mut Vec<T>,
) {
    // Deduplicate set of key-value pairs.
    buf1.clear();
    buf1.extend(keys.iter().copied().enumerate());
    buf1.sort_unstable_by_key(|(_, k)| *k);
    buf1.dedup_by_key(|(_, k)| *k);

    // Add ordered entries to secondary buffer.
    buf2.clear();
    for (i, _) in &*buf1 {
        let el = std::mem::take(&mut set[*i]);
        buf2.push(el);
    }

    // Now put them in place.
    set.clear();
    set.append(buf2);
}

impl Set {
    /// Flattens a multiset into a set.
    ///
    /// Every hereditary element of `set` is flattened as well, so that we're left with a valid
    /// [`Set`].
    pub fn from_mset(mut set: Mset) -> Self {
        let levels = Levels::new_mut(&mut set);

        let mut cur = Vec::new();
        let mut next = vec![0; levels.last().len()];

        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        let mut sets = BTreeMap::new();
        for level in levels.iter().rev().skip(1) {
            sets.clear();

            // Safety: Since we're modifying sets from bottom to top, we can ensure our pointers are
            // still valid.
            let iter = Levels::child_iter_gen(level, |s| unsafe { &*(s.cast_const()) }.card());
            for (i, range) in iter.enumerate() {
                // Deduplicate the set.
                unsafe {
                    let set = &mut **level.get_unchecked(i);
                    dedup_by(&mut set.0, next.get_unchecked(range), &mut buf1, &mut buf2);
                };

                let children: SmallVec<_> = buf1.iter().map(|(_, k)| *k).collect();
                let len = sets.len();
                match sets.entry(children) {
                    Entry::Vacant(entry) => {
                        entry.insert(len);
                        cur.push(len);
                    }
                    Entry::Occupied(entry) => {
                        cur.push(*entry.get());
                    }
                }
            }

            std::mem::swap(&mut cur, &mut next);
        }

        Self(set)
    }

    /// Transmutes an [`Mset`] into a [`Set`] without checking that there are no repeated elements.
    ///
    /// ## Safety
    ///
    /// You must guarantee that `set` is in fact a set. Doing otherwise breaks the type invariant
    /// for [`Set`].
    pub unsafe fn from_mset_unchecked(set: Mset) -> Self {
        Self(set)
    }
}

impl Mset {
    /// Checks whether the multiset is in fact a set. This property is checked hereditarily.
    pub fn is_set(&self) -> bool {
        // Subdivide the nodes of the set into levels.
        let mut levels = Vec::new();
        let mut last = vec![self];
        while !last.is_empty() {
            let mut cur = Vec::new();

            for &set in &last {
                for el in set {
                    cur.push(el);
                }
            }

            levels.push(last);
            last = cur;
        }

        // Given the sets from the next level (encoded as integers), finds encodings for the sets
        // in this level.
        let rank = levels.len();
        let mut next = vec![0; levels[rank - 1].len()];

        // Sets found on each level.
        // Each set gets assigned a unique integer.
        let mut sets = BTreeMap::new();
        for level in levels.into_iter().rev().skip(1) {
            sets.clear();
            let mut cur = Vec::with_capacity(level.len());

            let mut child = 0;
            for set in level {
                let mut el = SmallVec::new();
                for _ in 0..set.card() {
                    el.push(next[child]);
                    child += 1;
                }

                // Check for duplicates.
                el.sort_unstable();
                let el_len = el.len();
                el.dedup();
                if el.len() != el_len {
                    return false;
                }

                let len = sets.len();
                // Increase the count for each set.
                match sets.entry(el) {
                    Entry::Vacant(entry) => {
                        entry.insert(len);
                        cur.push(len);
                    }
                    Entry::Occupied(entry) => {
                        cur.push(*entry.get());
                    }
                }
            }

            next = cur;
        }

        true
    }

    /// Flattens a multiset into a set.
    ///
    /// See [`Set::from_mset`].
    pub fn into_set(self) -> Set {
        Set::from_mset(self)
    }

    /// Transmutes an [`Mset`] into a [`Set`] without checking that there are no repeated elements.
    ///
    /// ## Safety
    ///
    /// See [`Set::from_mset_unchecked`].
    pub unsafe fn into_set_unchecked(self) -> Set {
        Set::from_mset_unchecked(self)
    }

    pub unsafe fn as_set(&self) -> &Set {
        unsafe { &*(std::ptr::from_ref(self).cast()) }
    }
}

impl TryFrom<Mset> for Set {
    type Error = ();
    fn try_from(value: Mset) -> Result<Self, Self::Error> {
        if value.is_set() {
            Ok(Self(value))
        } else {
            Err(())
        }
    }
}

impl Set {
    /// Returns a reference to the underlying multiset.
    pub const fn mset(&self) -> &Mset {
        &self.0
    }

    /// The empty set Ø.
    pub const fn empty() -> Self {
        Self(Mset::empty())
    }

    /// Returns whether the multiset is finite.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Set cardinality.
    pub fn card(&self) -> usize {
        self.0.card()
    }

    /// An iterator over the elements of the [`Mset`].
    pub fn iter(&self) -> std::slice::Iter<Mset> {
        self.0.iter()
    }

    /// Finds the [`Ahu`] encoding for a set.
    pub fn ahu(&self) -> Ahu {
        self.0.ahu()
    }

    /// Set membership ∈.
    pub fn mem(&self, other: &Self) -> bool {
        self.0.mem(&other.0)
    }

    /// Subset ⊆.
    pub fn subset(&self, other: &Self) -> bool {
        self.0.subset(&other.0)
    }

    /// Mutable set insertion.
    pub fn insert_mut(&mut self, other: Self) {
        if !other.mem(self) {
            self.0.insert_mut(other.0);
        }
    }

    /// Mutable set insertion. Does not check whether the set being inserted is already in the set.
    ///
    /// ## Safety
    ///
    /// You must guarantee that `other` does not belong to `self`. Doing otherwise breaks the type
    /// invariant for [`Set`].
    pub unsafe fn insert_mut_unchecked(&mut self, other: Self) {
        self.0.insert_mut(other.0);
    }

    /// Set insertion x ∪ {y}.
    #[must_use]
    pub fn insert(mut self, other: Self) -> Self {
        self.insert_mut(other);
        self
    }

    /// Set insertion x ∪ {y}. Does not check whether the set being inserted is already in the set.
    ///
    /// ## Safety
    ///
    /// You must guarantee that `other` does not belong to `self`. Doing otherwise breaks the type
    /// invariant for [`Set`].
    pub unsafe fn insert_unchecked(mut self, other: Self) -> Self {
        self.insert_mut_unchecked(other);
        self
    }

    /// Set singleton {x}.
    pub fn singleton(self) -> Self {
        Self(self.0.singleton())
    }

    /// Set pair {x, y}.
    pub fn pair(self, other: Self) -> Self {
        self.singleton().insert(other)
    }

    /// Set union x ∪ y.
    pub fn union(self, other: Self) -> Self {
        (self.0.union(other.0)).into_set()
    }

    /// Set union ∪x.
    pub fn big_union(self) -> Self {
        self.0.big_union().into_set()
    }

    /// Mutable set specification.
    /*pub fn select_mut<P: FnMut(&Set) -> bool>(&mut self, pred: P) {
        self.0.select_mut(pred);
    }*/

    /// Set specification.
    /*  pub fn select<P: FnMut(&Mset) -> bool>(mut self, mut pred: P) -> Self {
        let mut i = 0;
        while i < self.card() {
            if pred(&self.0[i]) {
                i += 1;
            } else {
                self.0.swap_remove(i);
            }
        }

        self
    }*/

    /// Powerset 2^x.
    pub fn powerset(self) -> Self {
        Self(self.0.powerset())
    }

    /// The von Neumann rank of the set.
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// The von Neumann ordinal for n.
    pub fn nat(n: usize) -> Self {
        Self(Mset::nat(n))
    }

    /// The von Neumann hierarchy.
    pub fn neumann(n: usize) -> Self {
        Self(Mset::neumann(n))
    }
}
