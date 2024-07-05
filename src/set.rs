//! Hereditarily finite sets [`Set`].

use std::mem::ManuallyDrop;

use crate::prelude::*;

/// A set is a multiset that hereditarily has no duplicate elements.
///
/// ## Invariants
///
/// Every two elements in a [`Set`] must be distinct. Moreover, every element must satisfy this same
/// guarantee.
#[derive(Clone, PartialEq, Eq, PartialOrd)]
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

/// Transmute [`Vec<Set>`] into [`Vec<Mset>`].
fn cast_vec(vec: Vec<Set>) -> Vec<Mset> {
    let mut vec = ManuallyDrop::new(vec);
    unsafe { Vec::from_raw_parts(vec.as_mut_ptr().cast(), vec.len(), vec.capacity()) }
}

/// Orders and deduplicates a set based on the corresponding keys.
///
/// ## Safety
///
/// Both `set` and `keys` must have the same number of elements.
unsafe fn dedup_by<T: Default>(
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
        let el = std::mem::take(set.get_unchecked_mut(*i));
        buf2.push(el);
    }

    // Now put them in place.
    set.clear();
    set.append(buf2);
}

impl Set {
    /// Flattens a multiset into a set hereditarily.
    pub fn from_mset(mut set: Mset) -> Self {
        let levels = Levels::new_mut(&mut set);
        let mut cur = Vec::new();
        let mut next = vec![0; levels.last().len()];

        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        let mut sets = BTreeMap::new();
        for level in levels.iter().rev().skip(1) {
            sets.clear();
            cur.clear();

            // Safety: Since we're modifying sets from bottom to top, we can ensure our pointers are
            // still valid.
            let iter = Levels::child_iter_gen(level, |s| unsafe { &*(s.cast_const()) }.card());
            for (i, range) in iter.enumerate() {
                // Deduplicate the set.
                unsafe {
                    let set = &mut **level.get_unchecked(i);
                    dedup_by(&mut set.0, next.get_unchecked(range), &mut buf1, &mut buf2);
                };

                cur.push(btree_index(
                    &mut sets,
                    buf1.iter().map(|(_, k)| *k).collect::<SmallVec<_>>(),
                ));
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
    ///
    /// See also [`Self::into_set`].
    pub fn is_set(&self) -> bool {
        let levels = Levels::new(self);
        let mut cur = Vec::new();
        let mut next = vec![0; levels.last().len()];

        let mut sets = BTreeMap::new();
        for level in levels.iter().rev().skip(1) {
            sets.clear();
            cur.clear();

            for range in Levels::child_iter(level) {
                let slice = unsafe {
                    let slice = next.get_unchecked_mut(range);
                    slice.sort_unstable();
                    slice as &[_]
                };

                for i in 1..slice.len() {
                    if slice[i - 1] == slice[i] {
                        return false;
                    }
                }

                let children: SmallVec<usize> = slice.iter().copied().collect();
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

        true
    }

    /// Flattens a multiset into a set.
    ///
    /// See [`Set::from_mset`].
    pub fn into_set(self) -> Set {
        Set::from_mset(self)
    }

    /// Transmutes an [`Mset`] into a [`Set`] without checking the type invariants.
    ///
    /// ## Safety
    ///
    /// See [`Set::from_mset_unchecked`].
    pub unsafe fn into_set_unchecked(self) -> Set {
        Set::from_mset_unchecked(self)
    }

    /// Transmutes a [`Mset`] reference into a [`Set`] reference without checking the type
    /// invariants.
    ///
    /// ## Safety
    ///
    /// See [`Set::from_mset_unchecked`].
    pub unsafe fn as_set(&self) -> &Set {
        unsafe { &*(std::ptr::from_ref(self).cast()) }
    }

    /// Transmutes a [`Mset`] mutable reference into a [`Set`] mutable reference without checking
    /// the type invariants.
    ///
    /// ## Safety
    ///
    /// See [`Set::from_mset_unchecked`].
    pub unsafe fn as_set_mut(&mut self) -> &mut Set {
        unsafe { &mut *(std::ptr::from_mut(self).cast()) }
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

/// An auxiliary type to map [`Mset`] to [`Set`] within iterators.
pub struct Cast<I>(I);

impl Iterator for Cast<std::vec::IntoIter<Mset>> {
    type Item = Set;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(Set)
    }
}

impl<'a> Iterator for Cast<std::slice::Iter<'a, Mset>> {
    type Item = &'a Set;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|s| unsafe { s.as_set() })
    }
}

impl<'a> Iterator for Cast<std::slice::IterMut<'a, Mset>> {
    type Item = &'a mut Set;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|s| unsafe { s.as_set_mut() })
    }
}

impl IntoIterator for Set {
    type Item = Set;
    type IntoIter = Cast<std::vec::IntoIter<Mset>>;

    fn into_iter(self) -> Self::IntoIter {
        Cast(self.0.into_iter())
    }
}

impl<'a> IntoIterator for &'a Set {
    type Item = &'a Set;
    type IntoIter = Cast<std::slice::Iter<'a, Mset>>;

    fn into_iter(self) -> Self::IntoIter {
        Cast(self.0.iter())
    }
}

impl<'a> IntoIterator for &'a mut Set {
    type Item = &'a mut Set;
    type IntoIter = Cast<std::slice::IterMut<'a, Mset>>;

    fn into_iter(self) -> Self::IntoIter {
        Cast(self.0.iter_mut())
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

    /// Returns whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Clears the set.
    pub fn clear(&mut self) {
        self.0.clear()
    }

    /// Set cardinality.
    pub fn card(&self) -> usize {
        self.0.card()
    }

    /// An iterator over the elements of the [`Set`].
    pub fn iter(&self) -> Cast<std::slice::Iter<Mset>> {
        self.into_iter()
    }

    /// A mutable iterator over the elements of the [`Set`].
    pub fn iter_mut(&mut self) -> Cast<std::slice::IterMut<Mset>> {
        self.into_iter()
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

    /// Set union ∪x.
    pub fn big_union_vec(vec: Vec<Self>) -> Self {
        let union: Vec<Mset> = vec.into_iter().flatten().map(Into::into).collect();
        Mset(union).into_set()
    }

    /// Mutable set specification.
    pub fn select_mut<P: FnMut(&Set) -> bool>(&mut self, mut pred: P) {
        self.0.select_mut(|set| pred(unsafe { set.as_set() }));
    }

    /// Set specification.
    pub fn select<P: FnMut(&Set) -> bool>(mut self, pred: P) -> Self {
        self.select_mut(pred);
        self
    }

    /// Set intersection x ∩ y.
    ///
    /// This is a modified version of [`Mset::inter`].
    pub fn inter(self, other: Self) -> Self {
        let idx = self.card();
        let mut pair = self.0.pair(other.0);
        let levels = Levels::new(&pair);

        // The intersection of two empty sets is empty.
        let elements;
        if let Some(els) = levels.get(2) {
            elements = els;
        } else {
            return Self::empty();
        }

        // We store the indices of the sets in the intersection.
        let (mut next, mut indices) = levels.mod_ahu(3);

        let mut sets = BTreeMap::new();
        for (i, range) in Levels::child_iter(elements).enumerate() {
            let slice = unsafe {
                let slice = next.get_unchecked_mut(range);
                slice.sort_unstable();
                slice as &[_]
            };

            // Each entry stores the index where it's found within the first set.
            let children: SmallVec<_> = slice.iter().copied().collect();
            match sets.entry(children) {
                Entry::Vacant(entry) => {
                    if i < idx {
                        entry.insert(i);
                    }
                }
                Entry::Occupied(entry) => {
                    debug_assert!(i >= idx);
                    indices.push(entry.remove());
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

        Self(snd)
    }

    /*  /// Set intersection ∩x.
    pub fn big_inter(self) -> Option<Self> {
        Self(self.0.big_inter())
    }

    /// Set intersection ∩x.
    pub fn big_inter_vec(vec: Vec<Self>) -> Self {
        Self(Mset::big_inter(cast_vec(vec)))
    }*/

    /// Powerset 2^x.
    pub fn powerset(self) -> Self {
        Self(self.0.powerset())
    }

    /// The von Neumann rank of the set.
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// The von Neumann set encoding for n.
    pub fn nat(n: usize) -> Self {
        Self(Mset::nat(n))
    }

    /// The Zermelo set encoding for n.
    pub fn zermelo(n: usize) -> Self {
        Self(Mset::zermelo(n))
    }

    /// The von Neumann hierarchy.
    pub fn neumann(n: usize) -> Self {
        Self(Mset::neumann(n))
    }
}
