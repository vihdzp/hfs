//! Hereditarily finite sets [`Set`].

use crate::prelude::*;

/// A [hereditarily finite set](https://en.wikipedia.org/wiki/Hereditarily_finite_set), implemented
/// as a [`Mset`] where each multiset has no duplicate elements.
///
/// ## Invariants
///
/// These invariants should hold for any [`Set`]. **Unsafe code can perform optimizations contingent
/// on these.**
///
/// - Every two elements in a [`Set`] must be distinct.
/// - Any element in a [`Set`] must be a valid [`Set`] also.
#[derive(Clone, Default, AsRef, Display, Into, PartialEq, Eq, PartialOrd)]
#[repr(transparent)]
pub struct Set(Mset);

// -------------------- Basic traits -------------------- //

impl From<Set> for Vec<Set> {
    fn from(set: Set) -> Self {
        // Safety: elements of `Set` are valid for `Set`.
        unsafe { Set::cast_vec(set.0 .0) }
    }
}

/// Succintly writes a set as is stored in memory.
impl Debug for Set {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{:?}", self.mset())
    }
}

impl FromStr for Set {
    type Err = SetError;

    fn from_str(s: &str) -> Result<Self, SetError> {
        s.parse().map(Mset::flatten)
    }
}

// -------------------- Casting -------------------- //

/// Orders and deduplicates a set based on the corresponding keys.
///
/// - The first buffer is an intermediary buffer for calculations. It must be empty when this
/// function is called, but is emptied at the end of it.
/// - The second buffer is cleared within the function. At its output, it contains the set of
/// deduplicated keys with their indices in the original set.
///
/// ## Safety
///
/// Both `set` and `keys` must have the same number of elements.
unsafe fn dedup_by<T: Default, U: Ord + Copy>(
    set: &mut Vec<T>,
    keys: &[U],
    buf: &mut Vec<T>,
    buf_pairs: &mut Vec<(usize, U)>,
) {
    // Deduplicate set of key-value pairs.
    buf_pairs.clear();
    buf_pairs.extend(keys.iter().copied().enumerate());
    buf_pairs.sort_unstable_by_key(|(_, k)| *k);
    buf_pairs.dedup_by_key(|(_, k)| *k);

    // Add ordered entries to secondary buffer.
    for (i, _) in &*buf_pairs {
        let el = mem::take(set.get_unchecked_mut(*i));
        buf.push(el);
    }

    // Now put them in place.
    set.clear();
    set.append(buf);
}

impl Mset {
    /// Checks whether the multiset whose elements are given by an iterator is in fact a set. This
    /// property is checked hereditarily.
    ///
    /// See also [`Self::into_set`].
    #[must_use]
    pub fn is_set_iter<'a, I: IntoIterator<Item = &'a Self>>(iter: I) -> bool {
        Levels::new_iter(iter)
            .mod_ahu(
                0,
                BTreeMap::new(),
                |sets, slice, _| {
                    // Find duplicate elements.
                    slice.sort_unstable();
                    if consecutive_eq(&slice) {
                        return None;
                    }

                    let children: SmallVec<_> = slice.iter().copied().collect();
                    Some(btree_index(sets, children))
                },
                BTreeMap::clear,
            )
            .is_some()
    }

    /// Checks whether the multiset is in fact a set. This property is checked hereditarily.
    ///
    /// See also [`Self::into_set`].
    #[must_use]
    pub fn is_set(&self) -> bool {
        Self::is_set_iter(self)
    }

    /// Flattens a multiset into a set hereditarily.
    #[must_use]
    pub fn flatten(mut self) -> Set {
        let levels = Levels::new_mut(&mut self);
        let mut buf = Vec::new();
        let mut buf_pairs = Vec::new();

        // Safety: Since we're modifying sets from bottom to top, we can ensure our pointers are
        // still valid, as is our cardinality function.
        unsafe {
            levels.mod_ahu_gen(
                0,
                BTreeMap::new(),
                |sets, slice, set| {
                    // Deduplicate the set.
                    dedup_by(&mut (*set).0, slice, &mut buf, &mut buf_pairs);
                    let children: SmallVec<_> = buf_pairs.iter().map(|(_, k)| *k).collect();
                    Some(btree_index(sets, children))
                },
                BTreeMap::clear,
            );
        }

        // Safety: we just checked the invariant.
        unsafe { self.into_set_unchecked() }
    }

    /// Transmutes an [`Mset`] into a [`Set`] **without** checking the type invariants.
    ///
    /// ## Safety
    ///
    /// You must guarantee that the [`Mset`] satisfies the type invariants for [`Set`].
    #[must_use]
    pub unsafe fn into_set_unchecked(self) -> Set {
        Set(self)
    }

    /// Transmutes an [`Mset`] into a [`Set`], first checking the type invariants.
    ///
    /// To instead flatten the multiset, see [`Self::flatten`].
    #[must_use]
    pub fn into_set(self) -> Option<Set> {
        if self.is_set() {
            // Safety: we just checked the invariant.
            unsafe { Some(self.into_set_unchecked()) }
        } else {
            None
        }
    }

    /// Transmutes a [`Mset`] reference into a [`Set`] reference **without** checking the type
    /// invariants.
    ///
    /// ## Safety
    ///
    /// You must guarantee that the [`Mset`] satisfies the type invariants for [`Set`].
    #[must_use]
    pub unsafe fn as_set_unchecked(&self) -> &Set {
        // Safety: both types have the same layout.
        &*(ptr::from_ref(self).cast())
    }

    /// Transmutes a [`Mset`] reference into a [`Set`] reference, first checking the type
    /// invariants.
    #[must_use]
    pub fn as_set(&self) -> Option<&Set> {
        if self.is_set() {
            // Safety: we just checked the invariant.
            Some(unsafe { self.as_set_unchecked() })
        } else {
            None
        }
    }

    /// Transmutes a mutable [`Mset`] reference into a [`Set`] reference **without** checking the
    /// type invariants.
    ///
    /// ## Safety
    ///
    /// You must guarantee that the [`Mset`] satisfies the type invariants for [`Set`].
    #[must_use]
    pub unsafe fn as_set_mut_unchecked(&mut self) -> &mut Set {
        // Safety: both types have the same layout.
        &mut *(ptr::from_mut(self).cast())
    }

    /// Transmutes a mutable [`Mset`] reference into a [`Set`] reference, first checking the type
    /// invariants.
    #[must_use]
    pub fn as_set_mut(&mut self) -> Option<&mut Set> {
        if self.is_set() {
            // Safety: we just checked the invariant.
            Some(unsafe { self.as_set_mut_unchecked() })
        } else {
            None
        }
    }

    /// Converts `Vec<Set>` into `Vec<Mset>`.
    #[must_use]
    pub fn cast_vec(vec: Vec<Set>) -> Vec<Self> {
        // Safety: `Set` and `Mset` have the same layout.
        unsafe { crate::transmute_vec(vec) }
    }
}

impl Set {
    /// Returns a reference to the inner [`Mset`].
    #[must_use]
    pub const fn mset(&self) -> &Mset {
        &self.0
    }

    /// Returns whether an iterator over `Set` has no duplicate elements.
    ///
    /// This is analogous to [`Mset::is_set_iter`], but optimizes out the hereditary check.
    pub fn is_set_iter<'a, I: IntoIterator<Item = &'a Self>>(iter: I) -> bool {
        // We can optimize over `Mset::is_set` by not checking the no-duplicate property on things
        // we already know to be sets.
        let mut keys = Levels::new_iter(iter.into_iter().map(AsRef::as_ref)).ahu(0);
        keys.sort_unstable();
        consecutive_eq(&keys)
    }

    /// Returns whether a slice over `Set` has no duplicate elements.
    ///
    /// This is analogous to [`Mset::is_set`], but optimizes out the hereditary check.
    pub fn is_set(slice: &[Self]) -> bool {
        Self::is_set_iter(slice)
    }

    /// Converts `Vec<Set>` into `Set` by removing duplicate elements.
    ///
    /// This is analogous to [`Mset::flatten`], but optimizes out the hereditary check.
    pub fn flatten(mut vec: Vec<Self>) -> Self {
        // We can optimize over `Mset::flatten` by not checking the no-duplicate property on things
        // we already know to be sets.
        let mut keys = Levels::new_iter(vec.iter().map(AsRef::as_ref)).ahu(0);
        keys.sort_unstable();
        keys.dedup();

        for idx in keys.into_iter().rev() {
            vec.swap_remove(idx);
        }

        // Safety: We just removed duplicates.
        unsafe { Self::from_vec_unchecked(vec) }
    }

    /// Builds the set from a vector of sets. Does not deduplicate the set.
    ///
    /// ## Safety
    ///
    /// You must ensure any two sets in the vector are distinct.
    #[must_use]
    pub unsafe fn from_vec_unchecked(vec: Vec<Self>) -> Self {
        Self(Mset::cast_vec(vec).into())
    }

    /// Transmutes a `Vec<Set>` into a [`Set`], first checking the type invariants.
    ///
    /// To instead flatten the vector, see [`Self::flatten`].
    pub fn from_vec(vec: Vec<Self>) -> Option<Self> {
        if Self::is_set(&vec) {
            // Safety: we just checked the invariant.
            unsafe { Some(Self::from_vec_unchecked(vec)) }
        } else {
            None
        }
    }

    /// Converts `Vec<Mset>` into `Vec<Set>`.
    ///
    /// ## Safety
    ///
    /// You must guarantee that the [`Mset`] satisfy the type invariants for [`Set`].
    #[must_use]
    pub unsafe fn cast_vec(vec: Vec<Mset>) -> Vec<Self> {
        crate::transmute_vec(vec)
    }
}

// -------------------- Iterators -------------------- //

/// An auxiliary type to map [`Mset`] to [`Set`] within iterators.
///
/// ## Invariants
///
/// This can only be used with iterators coming from a [`Set`]. Note that this is guaranteed by all
/// of the public methods that generate this type.
pub struct Cast<I>(I);

impl Iterator for Cast<std::vec::IntoIter<Mset>> {
    type Item = Set;

    fn next(&mut self) -> Option<Self::Item> {
        // Safety: we're iterating over a set.
        self.0.next().map(|s| unsafe { s.into_set_unchecked() })
    }
}

impl<'a> Iterator for Cast<slice::Iter<'a, Mset>> {
    type Item = &'a Set;

    fn next(&mut self) -> Option<Self::Item> {
        // Safety: we're iterating over a set.
        self.0.next().map(|s| unsafe { s.as_set_unchecked() })
    }
}

impl<'a> Iterator for Cast<slice::IterMut<'a, Mset>> {
    type Item = &'a mut Set;

    fn next(&mut self) -> Option<Self::Item> {
        // Safety: we're iterating over a set.
        self.0.next().map(|s| unsafe { s.as_set_mut_unchecked() })
    }
}

impl IntoIterator for Set {
    type Item = Set;
    type IntoIter = Cast<std::vec::IntoIter<Mset>>;

    fn into_iter(self) -> Self::IntoIter {
        Cast(self.0.into_iter())
    }
}

// The `iter` function is defined in SetTrait.
#[allow(clippy::into_iter_without_iter)]
impl<'a> IntoIterator for &'a Set {
    type Item = &'a Set;
    type IntoIter = Cast<slice::Iter<'a, Mset>>;

    fn into_iter(self) -> Self::IntoIter {
        Cast(self.0.iter())
    }
}

impl<'a> IntoIterator for &'a mut Set {
    type Item = &'a mut Set;
    type IntoIter = Cast<slice::IterMut<'a, Mset>>;

    fn into_iter(self) -> Self::IntoIter {
        Cast(self.0.iter_mut())
    }
}

// -------------------- SetTrait -------------------- //

impl crate::Seal for Set {}

impl SetTrait for Set {
    // -------------------- Basic methods -------------------- //

    fn as_slice(&self) -> &[Self] {
        let slice = self.0.as_slice();
        // Safety: `Set` and `Mset` have the same layout.
        unsafe { slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    unsafe fn _as_mut_slice(&mut self) -> &mut [Self] {
        let slice = self.0.as_mut_slice();
        // Safety: `Set` and `Mset` have the same layout.
        slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len())
    }

    fn _flatten_vec(vec: Vec<Self>) -> Self {
        Self::flatten(vec)
    }

    fn as_vec(&self) -> &Vec<Mset> {
        self.0.as_vec()
    }

    unsafe fn _as_mut_vec(&mut self) -> &mut Vec<Mset> {
        self.0._as_mut_vec()
    }

    // -------------------- Constructions -------------------- //

    fn empty() -> Self {
        Self(Mset::empty())
    }

    fn with_capacity(capacity: usize) -> Self {
        Self(Mset::with_capacity(capacity))
    }

    fn singleton(self) -> Self {
        Self(self.0.singleton())
    }

    fn into_singleton(self) -> Option<Self> {
        self.0.into_singleton().map(Self)
    }

    fn insert_mut(&mut self, set: Self) {
        self.try_insert(set);
    }

    fn select_mut<P: FnMut(&Set) -> bool>(&mut self, mut pred: P) {
        self.0
            // Safety: we're iterating over a set.
            .select_mut(|set| pred(unsafe { set.as_set_unchecked() }));
    }

    fn count(&self, other: &Self) -> usize {
        usize::from(self.contains(other))
    }

    fn sum_vec(vec: Vec<Self>) -> Self {
        Self::flatten(vec.into_iter().flatten().collect())
    }

    fn union_vec(vec: Vec<Self>) -> Self {
        Self::sum_vec(vec)
    }

    fn inter_vec(mut vec: Vec<Self>) -> Option<Self> {
        // Check for trivial cases.
        match vec.len() {
            0 => return None,
            1 => return Some(vec.pop().unwrap()),
            _ => {}
        }

        let levels = Levels::new_iter(vec.iter().map(AsRef::as_ref));
        let next = levels.ahu(1);
        // Safety: the length of `next` is exactly the sum of cardinalities in the first level.
        let mut iter = unsafe { levels.children_slice(0, &next) };

        // Each entry stores the index where it's found within the first set, and a boolean for
        // whether it's been seen in every other set.
        //
        // Safety: we already know there's at least 2 sets.
        let fst = unsafe { iter.next().unwrap_unchecked() };
        let mut sets = BTreeMap::new();
        for (i, set) in fst.iter().enumerate() {
            if sets.insert(*set, (i, false)).is_some() {
                // Safety: sets don't have duplicates.
                unsafe { hint::unreachable_unchecked() }
            }
        }

        // Look for appearances in other sets.
        for slice in iter {
            for set in slice {
                match sets.entry(*set) {
                    Entry::Vacant(_) => {}
                    Entry::Occupied(mut entry) => entry.get_mut().1 = true,
                }
            }

            // Update counts.
            sets.retain(|_, (_, count)| {
                let retain = *count;
                *count = false;
                retain
            });
        }

        // Take elements from the first set, reuse some other set as a buffer.
        let mut fst = vec.swap_remove(0);
        let mut snd = vec.swap_remove(0);
        snd.clear();

        for (i, _) in sets.into_values() {
            // Safety: all the indices we built are valid for the first set.
            let set = mem::take(unsafe { fst._as_mut_slice().get_unchecked_mut(i) });
            snd.insert_mut(set);
        }

        Some(snd)
    }

    fn powerset(self) -> Self {
        Self(self.0.powerset())
    }

    fn nat(n: usize) -> Self {
        Self(Mset::nat(n))
    }

    fn zermelo(n: usize) -> Self {
        Self(Mset::zermelo(n))
    }

    fn neumann(n: usize) -> Self {
        Self(Mset::neumann(n))
    }

    fn into_choose(mut self) -> Option<Self> {
        self.0 .0.pop().map(
            // Safety: we're choosing a set.
            |s| unsafe { s.into_set_unchecked() },
        )
    }

    fn choose_uniq(&self) -> Option<&Self> {
        self.0.choose_uniq().map(
            // Safety: we're choosing a set.
            |s| unsafe { s.as_set_unchecked() },
        )
    }

    fn into_choose_uniq(self) -> Option<Self> {
        self.0.into_choose_uniq().map(
            // Safety: we're choosing a set.
            |s| unsafe { s.into_set_unchecked() },
        )
    }

    // -------------------- Relations -------------------- //

    /*
    fn disjoint_pairwise<'a, I: IntoIterator<Item = &'a Self>>(iter: I) -> bool {
        // Empty families are disjoint.
        let levels;
        if let Some(lev) = Levels::init_iter(iter.into_iter().map(AsRef::as_ref)) {
            levels = lev.fill();
        } else {
            return true;
        }

        // Empty sets are disjoint.
        let elements;
        if let Some(el) = levels.get(1) {
            elements = el;
        } else {
            return true;
        }

        let mut cur = Vec::new();
        let mut next = Vec::new();
        let mut sets = BTreeMap::new();

        // Compute AHU encodings for all but the elements of the union.
        for level in levels.iter().skip(2).rev() {
            sets.clear();

            // Safety: the length of `next` is exactly the sum of cardinalities in `level`.
            unsafe {
                Levels::step_ahu(level, &mut cur, &mut next, |slice, _| {
                    slice.sort_unstable();
                    let children = slice.iter().copied().collect::<SmallVec<_>>();
                    Some(btree_index(&mut sets, children))
                });
            }

            mem::swap(&mut cur, &mut next);
        }

        // Compute the encodings for the union. Return whether we find anything twice.
        let mut dummy: Vec<()> = Vec::new();
        sets.clear();

        // Safety: the length of next is exactly the sum of cardinalities in `elements`.
        unsafe {
            Levels::step_ahu(elements, &mut dummy, &mut next, |slice, _| {
                slice.sort_unstable();
                let children = slice.iter().copied().collect::<SmallVec<_>>();

                // The values don't matter, but we recycle our BTreeMap instead of creating a new
                // BTreeSet.
                if sets.insert(children, 0).is_some() {
                    None
                } else {
                    Some(())
                }
            })
        }
    }

    fn disjoint_iter<'a, I: IntoIterator<Item = &'a Self>>(_iter: I) -> bool {
        todo!()
    }
    */
}

// -------------------- Set specific -------------------- //

impl Set {
    /// The set as a mutable slice.
    ///
    /// ## Safety
    ///
    /// You must preserve the type invariants for [`Set`]. In particular, you can't make two
    /// elements equal.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [Self] {
        self._as_mut_slice()
    }

    /// A mutable reference to the inner vector.
    ///
    /// ## Safety
    ///
    /// You must preserve the type invariants for [`Set`]. In particular, you can't make two
    /// elements equal.
    pub unsafe fn as_mut_vec(&mut self) -> &mut Vec<Mset> {
        &mut self.0 .0
    }

    /// Mutably iterate over the elements of the set.
    ///
    /// ## Safety
    ///
    /// You must preserve the type invariants for [`Set`]. In particular, you can't make two
    /// elements equal.
    pub unsafe fn iter_mut(&mut self) -> slice::IterMut<Self> {
        self.as_mut_slice().iter_mut()
    }

    /// In-place set insertion x ∪ {y}. Does not check whether the set being inserted is already in
    /// the set.
    ///
    /// ## Safety
    ///
    /// You must guarantee that `other` does not belong to `self`. Doing otherwise breaks the type
    /// invariants for [`Set`].
    pub unsafe fn insert_mut_unchecked(&mut self, other: Self) {
        self.0.insert_mut(other.0);
    }

    /// Set insertion x ∪ {y}. Does not check whether the set being inserted is already in the set.
    ///
    /// ## Safety
    ///
    /// You must guarantee that `other` does not belong to `self`. Doing otherwise breaks the type
    /// invariants for [`Set`].
    #[must_use]
    pub unsafe fn insert_unchecked(mut self, other: Self) -> Self {
        self.insert_mut_unchecked(other);
        self
    }

    /// Inserts an element into a set in place. Returns whether the size of the set changed.
    pub fn try_insert(&mut self, set: Self) -> bool {
        let res = !self.contains(&set);
        if res {
            // Safety: we just performed the relevant check.
            unsafe {
                self.insert_mut_unchecked(set);
            }
        }
        res
    }

    /// Set pair {x, y} = {x} + {y}. Does not check whether x ≠ y.
    ///
    /// ## Safety
    ///
    /// You must guarantee that both sets are distinct.
    #[must_use]
    pub unsafe fn pair_unchecked(self, other: Self) -> Self {
        Self(self.0.pair(other.0))
    }

    /// [Replaces](https://en.wikipedia.org/wiki/Axiom_schema_of_replacement) the elements in a set
    /// by applying a function. Does not verify that the mapped elements are distinct.
    ///
    /// ## Safety
    ///
    /// You must guarantee that the function does not yield the same output for two distinct
    /// elements of the set.
    #[must_use]
    pub unsafe fn replace_unchecked<F: FnMut(&Self) -> Self>(&self, mut func: F) -> Self {
        Self(self.0.replace(|set| func(set.as_set_unchecked()).0))
    }

    /// [Replaces](https://en.wikipedia.org/wiki/Axiom_schema_of_replacement) the elements in a set
    /// by applying a function. Does not verify that the mapped elements are distinct.
    ///
    /// ## Safety
    ///
    /// You must guarantee that the function does not yield the same output for two distinct
    /// elements of the set.
    #[must_use]
    pub unsafe fn into_replace_unchecked<F: FnMut(Self) -> Self>(self, mut func: F) -> Self {
        Self(self.0.into_replace(|set| func(set.into_set_unchecked()).0))
    }
}

// -------------------- Ordered pairs -------------------- //

/// Encodes the elements stored in a [Kuratowski
/// pair](https://en.wikipedia.org/wiki/Ordered_pair#Kuratowski's_definition).
///
/// A pair (x, x) is equal to {{x}}, which means x is not allocated twice. This structure thus
/// avoids needless clones.
pub enum Kpair<T> {
    /// A pair (x, x).
    Same(T),
    /// A pair (x, y) with x ≠ y.
    Distinct(T, T),
}

impl<T> Kpair<T> {
    /// Initializes a new [`Kpair`].
    pub fn new(x: T, y: T) -> Self
    where
        T: PartialEq,
    {
        if x == y {
            Self::Same(x)
        } else {
            Self::Distinct(x, y)
        }
    }

    /// Returns the first entry.
    pub fn into_fst(self) -> T {
        match self {
            Self::Same(x) | Self::Distinct(x, _) => x,
        }
    }

    /// Returns the second entry.
    pub fn into_snd(self) -> T {
        match self {
            Self::Same(x) | Self::Distinct(_, x) => x,
        }
    }

    /// Returns a reference to the first entry.
    pub const fn fst(&self) -> &T {
        match self {
            Self::Same(x) | Self::Distinct(x, _) => x,
        }
    }

    /// Returns a reference to the second entry.
    pub const fn snd(&self) -> &T {
        match self {
            Self::Same(x) | Self::Distinct(_, x) => x,
        }
    }

    /// Converts a [`Kpair`] into a standard pair. Note that this might require a clone operation.
    pub fn into_pair(self) -> (T, T)
    where
        T: Clone,
    {
        match self {
            Self::Same(x) => (x.clone(), x),
            Self::Distinct(x, y) => (x, y),
        }
    }

    /// Converts a [`Kpair`] into a standard pair.
    pub fn pair(self) -> (T, T)
    where
        T: Copy,
    {
        self.into_pair()
    }
}

impl Set {
    /// A [Kuratowski pair](https://en.wikipedia.org/wiki/Ordered_pair#Kuratowski's_definition) (x,
    /// x) = {{x}}.
    #[must_use]
    pub fn id_kpair(self) -> Self {
        self.singleton().singleton()
    }

    /// A [Kuratowski pair](https://en.wikipedia.org/wiki/Ordered_pair#Kuratowski's_definition) (x,
    /// y) = {{x}, {x, y}}. Does not check whether x ≠ y.
    ///
    /// ## Safety
    ///
    /// You must ensure that both sets are distinct.
    #[must_use]
    pub unsafe fn kpair_unchecked(self, other: Self) -> Self {
        let (x, y) = (self.0, other.0);
        // Safety: if x ≠ y, then {x} ≠ {x, y}.
        x.clone().singleton().pair(x.pair(y)).into_set_unchecked()
    }

    /// A [Kuratowski pair](https://en.wikipedia.org/wiki/Ordered_pair#Kuratowski's_definition) (x,
    /// y) = {{x}, {x, y}}.
    #[must_use]
    pub fn kpair(self, other: Self) -> Self {
        if self == other {
            self.id_kpair()
        } else {
            // Safety: we just performed the relevant check.
            unsafe { self.kpair_unchecked(other) }
        }
    }

    /// Decomposes a Kuratowski pair.
    #[must_use]
    pub fn ksplit(&self) -> Option<Kpair<&Self>> {
        match self.as_slice() {
            [set] => match set.as_slice() {
                [a] => Some(Kpair::Same(a)),
                _ => None,
            },
            [fst, snd] => match (fst.as_slice(), snd.as_slice()) {
                ([a], [b, c]) | ([b, c], [a]) => {
                    if a == b {
                        Some(Kpair::Distinct(a, c))
                    } else if a == c {
                        Some(Kpair::Distinct(a, b))
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Decomposes a Kuratowski pair.
    #[must_use]
    pub fn into_ksplit(mut self) -> Option<Kpair<Self>> {
        // Safety: our usage of `as_mut_slice` causes no issues, as the set is dropped and discarded
        // when the method returns.
        unsafe {
            match self.as_mut_slice() {
                [set] => match set.as_mut_slice() {
                    [a] => Some(Kpair::Same(mem::take(a))),
                    _ => None,
                },
                [fst, snd] => match (fst.as_mut_slice(), snd.as_mut_slice()) {
                    ([a], [b, c]) | ([b, c], [a]) => {
                        if a == b {
                            Some(Kpair::Distinct(mem::take(a), mem::take(c)))
                        } else if a == c {
                            Some(Kpair::Distinct(mem::take(a), mem::take(b)))
                        } else {
                            None
                        }
                    }
                    _ => None,
                },
                _ => None,
            }
        }
    }

    /// Tagged or disjoint union.
    ///
    /// See [`Self::tag_union`].
    pub fn tag_union_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        let mut union = Self::empty();
        for set in iter {
            // Safety: since our original sets and the elements in them were distinct, so are our
            // pairs.
            unsafe {
                for element in &set.as_slice()[1..] {
                    union.insert_mut_unchecked(set.clone().kpair(element.clone()));
                }

                // Reuse `set` allocation.
                if let Some(fst) = set.as_slice().first().cloned() {
                    union.insert_mut_unchecked(set.kpair(fst));
                }
            }
        }

        union
    }

    /// Tagged or disjoint union over a vector.
    ///
    /// See [`Self::tag_union`].
    #[must_use]
    pub fn tag_union_vec(vec: Vec<Self>) -> Self {
        Self::tag_union_iter(vec)
    }

    /// Tagged or disjoint union.
    ///
    /// This returns the set of all pairs (x, y), where x is either of the two sets, and y is an
    /// element of it.
    ///
    /// Whereas the usual union can vary in cardinality, the tagged union of two sets always adds
    /// their cardinalities.
    #[must_use]
    pub fn tag_union(self, other: Self) -> Self {
        Self::tag_union_iter([self, other])
    }

    /// Cartesian product of sets.
    #[must_use]
    pub fn prod(mut self, mut other: Self) -> Self {
        // Ensure `self` is the smallest set.
        let a = self.card();
        let b = other.card();
        if b < a {
            mem::swap(&mut other, &mut self);
        }

        let mut prod = Self::with_capacity(a * b);
        // Safety: these are ordered pairs of distinct pairs of elements.
        unsafe {
            for (i, fst) in self.iter().enumerate() {
                for (j, snd) in other.iter().enumerate() {
                    if i != j {
                        prod.insert_mut_unchecked(fst.clone().kpair(snd.clone()));
                    }
                }
            }

            // Re-use allocations.
            for (fst, snd) in self.into_iter().zip(other.into_iter()) {
                prod.insert_mut_unchecked(fst.kpair(snd));
            }
        }

        prod
    }
}

// -------------------- Functions -------------------- //

impl Set {
    /// Generalizes [`Self::dom`] and [`Self::range`].
    fn dom_range<F: FnMut(Kpair<Self>) -> Self>(self, mut entry: F) -> Option<Vec<Self>> {
        let mut vec = self.into_vec();
        for set in &mut vec {
            let s = mem::take(set);
            *set = entry(s.into_ksplit()?);
        }
        Some(vec)
    }

    /// The domain of a relation.
    pub fn dom(self) -> Option<Self> {
        self.dom_range(Kpair::into_fst).map(Self::flatten)
    }

    /// The range of a relation or function.
    pub fn range(self) -> Option<Self> {
        self.dom_range(Kpair::into_snd).map(Self::flatten)
    }

    /// The domain of a function. This optimizes over [`Self::dom`] by assuming that every value in
    /// the domain is mapped to a unique output.
    ///
    /// ## Safety
    ///
    /// You must guarantee that, if `self` is a relation, then it is also a valid function.
    pub unsafe fn dom_func(self) -> Option<Self> {
        self.dom_range(Kpair::into_fst)
            .map(|vec| Self::from_vec_unchecked(vec))
    }

    /// Evaluates a function at a set. Returns `None` if the set is not in the domain.
    ///
    /// If `self` is not a function, the result will almost definitely be garbage.
    #[must_use]
    pub fn eval(&self, set: &Self) -> Option<&Self> {
        let mut cmp = Compare::new(set.mset());
        self.iter().map_while(Set::ksplit).find_map(|pair| {
            if cmp.eq(pair.fst().mset()) {
                Some(pair.into_snd())
            } else {
                None
            }
        })
    }

    /// Evaluates a function at a set. Returns `None` if the set is not in the domain.
    ///
    /// If `self` is not a function, the result will almost definitely be garbage.
    #[must_use]
    pub fn into_eval(self, set: &Self) -> Option<Self> {
        let mut cmp = Compare::new(set.mset());
        self.into_iter()
            .map_while(Set::into_ksplit)
            .find_map(|pair| {
                if cmp.eq(pair.fst().mset()) {
                    Some(pair.into_snd())
                } else {
                    None
                }
            })
    }

    /// Returns the identity function with domain `self`.
    #[must_use]
    pub fn id_func(self) -> Self {
        // Safety: all elements were originally sets and are distinct.
        unsafe { Self::from_vec_unchecked(self.into_iter().map(Set::id_kpair).collect()) }
    }

    /// Returns the constant function with domain `self` and value `cst`.
    #[must_use]
    pub fn const_func(self, cst: Set) -> Self {
        let mut func = Set::with_capacity(self.card());
        let mut vec = self.into_vec();

        // Safety: all these pairs are distinct.
        unsafe {
            for set in vec.drain(1..) {
                func.insert_mut_unchecked(set.kpair(cst.clone()));
            }

            // Reuse `cst`.
            if let Some(fst) = vec.pop() {
                func.insert_mut_unchecked(fst.kpair(cst));
            }
        }

        func
    }

    /// Set of functions between two sets x → y.
    ///
    /// ## Panics
    ///
    /// This function will panic if you attempt to create a set that's too large. Note that this is
    /// quite easy to do, as |x → y| = |y|<sup>|x|</sup>.
    #[must_use]
    pub fn func(self, mut other: Self) -> Self {
        let dom_card = self.card();
        let cod_card = other.card();

        // The empty function is the only function with domain Ø.
        if dom_card == 0 {
            return Self::empty().singleton();
        }

        match cod_card {
            // No other function has codomain Ø.
            0 => return Self::empty(),
            // There is only one function into a singleton.
            1 => {
                // Safety: this is a singleton.
                return Self::const_func(self, unsafe {
                    other.into_singleton().unwrap_unchecked()
                })
                .singleton();
            }
            _ => {}
        }

        let size = cod_card.pow(dom_card.try_into().expect("domain too large"));
        let mut funcs = Self::with_capacity(size);

        // The indices in `other` into which we map the elements in `self`.
        let mut indices = vec![0; dom_card];
        for _ in 1..size {
            // Update indices.
            let mut idx = dom_card - 1;
            loop {
                // Safety: our indices essentially function as a base `cod_card` expansion. We exit
                // the outer loop just as all values are maxed out.
                unsafe {
                    let index = indices.get_unchecked_mut(idx);
                    *index += 1;
                    if *index == cod_card {
                        *index = 0;
                    } else {
                        break;
                    }

                    idx = idx.unchecked_sub(1);
                }
            }

            // Safety: all pairs within a single function have distinct first entries, and are thus
            // distinct. All the functions we build in general are distinct.
            unsafe {
                let mut func = Self::with_capacity(dom_card);
                for (i, &j) in indices.iter().enumerate() {
                    func.insert_mut_unchecked(
                        self.as_slice()
                            .get_unchecked(i)
                            .clone()
                            .kpair(other.as_slice().get_unchecked(j).clone()),
                    );
                }

                funcs.insert_mut_unchecked(func);
            }
        }

        // Reuse `self`.
        // Safety: same as above.
        unsafe {
            funcs.insert_mut_unchecked(
                self.const_func(mem::take(other.as_mut_slice().get_unchecked_mut(0))),
            );
        }
        funcs
    }
}
