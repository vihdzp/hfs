//! Hereditarily finite sets [`Set`].

use crate::prelude::*;

/// A [hereditarily finite set](https://en.wikipedia.org/wiki/Hereditarily_finite_set), implemented
/// as a [`Mset`] where each multiset has no duplicate elements.
///
/// ## Invariants
///
/// These invariants should hold for any [`Set`]. **Unsafe code performs optimizations contingent on
/// these.**
///
/// - Every two elements in a [`Set`] must be distinct.
/// - Any element in a [`Set`] must be a valid [`Set`] also.
#[derive(Clone, Default, Eq)]
#[repr(transparent)]
pub struct Set(Mset);

// -------------------- Basic traits -------------------- //

impl AsRef<Mset> for Set {
    fn as_ref(&self) -> &Mset {
        &self.0
    }
}

impl From<Set> for Mset {
    fn from(set: Set) -> Self {
        set.0
    }
}

impl From<Set> for Vec<Set> {
    fn from(set: Set) -> Self {
        // Safety: elements of `Set` are valid for `Set`.
        unsafe { Mset::cast_vec(set.0 .0) }
    }
}

/// Succintly writes a multiset as stored in memory.
impl Debug for Set {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{:?}", self.mset())
    }
}

/// Displays a multiset in canonical roster notation.
impl Display for Set {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self.mset())
    }
}

impl FromStr for Set {
    type Err = SetError;

    fn from_str(s: &str) -> Result<Self, SetError> {
        s.parse().map(Mset::into_set)
    }
}

// -------------------- Casting -------------------- //

/// Transmute a vector of one type into a vector of another type.
///
/// ## Safety
///
/// The types `T` and `U` must be transmutable into each other. In particular, they must have the
/// same size and alignment.
unsafe fn transmute_vec<T, U>(vec: Vec<T>) -> Vec<U> {
    assert_eq!(mem::size_of::<T>(), mem::size_of::<U>());
    assert_eq!(mem::align_of::<T>(), mem::align_of::<U>());

    let mut vec = mem::ManuallyDrop::new(vec);
    Vec::from_raw_parts(vec.as_mut_ptr().cast(), vec.len(), vec.capacity())
}

/// Orders and deduplicates a set based on the corresponding keys.
///
/// The first buffer is an intermediary buffer for calculations. It must be empty when this function
/// is called, but is emptied at the end of it.
///
/// The second buffer is cleared within the function. At its output, it contains the set of
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
    /// Flattens a multiset into a set hereditarily.
    #[must_use]
    pub fn into_set(mut self) -> Set {
        let levels = Levels::init(std::ptr::from_mut(&mut self)).fill_mut();
        let mut buf = Vec::new();
        let mut buf_pairs = Vec::new();

        // Safety: Since we're modifying sets from bottom to top, we can ensure our pointers are
        // still valid, as does our cardinality function.
        unsafe {
            levels.mod_ahu_gen(
                1,
                BTreeMap::new(),
                |s| (*s.cast_const()).card(),
                |sets, slice, &set| {
                    // Deduplicate the set.
                    dedup_by(&mut (*set).0, slice, &mut buf, &mut buf_pairs);
                    let children: SmallVec<_> = buf_pairs.iter().map(|(_, k)| *k).collect();
                    Some(btree_index(sets, children))
                },
                BTreeMap::clear,
            );
        }

        Set(self)
    }

    /// Checks whether the multiset is in fact a set. This property is checked hereditarily.
    ///
    /// See also [`Self::into_set`].
    #[must_use]
    pub fn is_set(&self) -> bool {
        Levels::init(self)
            .fill()
            .mod_ahu(
                1,
                BTreeMap::new(),
                |sets, slice, _| {
                    // Find duplicate elements.
                    slice.sort_unstable();
                    if has_consecutive(slice) {
                        return None;
                    }

                    let children: SmallVec<_> = slice.iter().copied().collect();
                    Some(btree_index(sets, children))
                },
                BTreeMap::clear,
            )
            .is_some()
    }

    /// Transmutes an [`Mset`] into a [`Set`], first checking the type invariants.
    #[must_use]
    pub fn into_set_checked(self) -> Option<Set> {
        if self.is_set() {
            Some(Set(self))
        } else {
            None
        }
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

    /// Transmutes a [`Mset`] reference into a [`Set`] reference, first checking the type
    /// invariants.
    #[must_use]
    pub fn as_set_checked(&self) -> Option<&Set> {
        if self.is_set() {
            // Safety: both types have the same layout, and we just checked the invariant.
            Some(unsafe { &*(std::ptr::from_ref(self).cast()) })
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
        self.as_set_checked().unwrap_unchecked()
    }

    /// Transmutes a mutable [`Mset`] reference into a [`Set`] reference, first checking the type
    /// invariants.
    #[must_use]
    pub fn as_set_mut_checked(&mut self) -> Option<&mut Set> {
        if self.is_set() {
            // Safety: both types have the same layout, and we just checked the invariant.
            Some(unsafe { &mut *(std::ptr::from_mut(self).cast()) })
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
        self.as_set_mut_checked().unwrap_unchecked()
    }

    /// Converts `Vec<Mset>` into `Vec<Set>`.
    ///
    /// ## Safety
    ///
    /// You must guarantee that the [`Mset`] satisfy the type invariants for [`Set`].
    #[must_use]
    pub unsafe fn cast_vec(vec: Vec<Self>) -> Vec<Set> {
        transmute_vec(vec)
    }
}

impl Set {
    /// Converts `Vec<Set>` into `Vec<Mset>`.
    #[must_use]
    pub fn cast_vec(vec: Vec<Self>) -> Vec<Mset> {
        // Safety: `Set` and `Mset` have the same layout.
        unsafe { transmute_vec(vec) }
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
        self.0.next().map(Set)
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
        // Safety: Set and Mset have the same layout.
        unsafe { slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    unsafe fn _as_mut_slice(&mut self) -> &mut [Self] {
        let slice = self.0.as_mut_slice();
        // Safety: Set and Mset have the same layout.
        slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len())
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

    fn singleton(self) -> Self {
        Self(self.0.singleton())
    }

    fn insert_mut(&mut self, set: Self) {
        if !self.contains(&set) {
            self.0.insert_mut(set.0);
        }
    }

    fn select_mut<P: FnMut(&Set) -> bool>(&mut self, mut pred: P) {
        self.0
            // Safety: we're iterating over a set.
            .select_mut(|set| pred(unsafe { set.as_set_unchecked() }));
    }

    fn sum_vec(mut vec: Vec<Self>) -> Self {
        let levels;
        if let Some(lev) = Levels::init_iter(vec.iter().map(AsRef::as_ref)) {
            levels = lev;
        } else {
            return Self::empty();
        }

        let keys = levels.fill().ahu(0);
        // Safety: `keys` has as many elements as `vec`.
        unsafe {
            dedup_by(&mut vec, &keys, &mut Vec::new(), &mut Vec::new());
        }

        Self(Mset(Self::cast_vec(vec)))
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
        let levels = Levels::init_iter(vec.iter().map(AsRef::as_ref))
            .unwrap()
            .fill();

        let next = levels.ahu(1);
        // Safety: the length of `next` is exactly the sum of cardinalities in the first level.
        let mut iter = unsafe { Levels::child_iter(levels.first(), &next) };

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
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().1 = true;
                    }
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

    // -------------------- Relations -------------------- //

    unsafe fn _levels_subset(fst: &Levels<&Mset>, snd: &Levels<&Mset>) -> bool {
        fst.both_ahu(
            snd,
            // Remove found sets, return if one isn't found.
            |sets, children| sets.remove(&children),
            // Add found sets. No set can be duplicated.
            |sets, children| {
                let len = sets.len();
                if sets.insert(children, len).is_some() {
                    // Safety: sets don't have duplicates.
                    unsafe { hint::unreachable_unchecked() }
                }
                len
            },
        )
    }

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
}

impl Set {
    /// Returns a reference to the inner [`Mset`].
    #[must_use]
    pub const fn mset(&self) -> &Mset {
        &self.0
    }

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
}

// -------------------- Constructions -------------------- //

impl Set {
    /// Kuratowski pair (x, y).
    #[must_use]
    pub fn kpair(self, other: Self) -> Self {
        self.clone().singleton().pair(self.pair(other))
    }

    /// Decomposes a Kuratowski pair.
    #[must_use]
    pub fn ksplit(&self) -> Option<(&Self, &Self)> {
        match self.as_slice() {
            [set] => match set.as_slice() {
                [a] => Some((a, a)),
                _ => None,
            },
            [fst, snd] => match (fst.as_slice(), snd.as_slice()) {
                ([a], [b, c]) | ([b, c], [a]) => {
                    if a == b {
                        Some((a, c))
                    } else if a == c {
                        Some((a, b))
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
    pub fn into_ksplit(mut self) -> Option<(Self, Self)> {
        // Safety: our usage of `as_mut_slice` causes no issues, as the set is dropped and discarded
        // when the method returns.
        unsafe {
            match self.as_mut_slice() {
                [set] => match set.as_mut_slice() {
                    [a] => Some((a.clone(), mem::take(a))),
                    _ => None,
                },
                [fst, snd] => match (fst.as_mut_slice(), snd.as_mut_slice()) {
                    ([a], [b, c]) | ([b, c], [a]) => {
                        if a == b {
                            Some((mem::take(a), mem::take(c)))
                        } else if a == c {
                            Some((mem::take(a), mem::take(b)))
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
}

/*
#[cfg(test)]
mod tests {
    use super::*;

    /// Verify round-trip between set and string.
    fn roundtrip(set: &Set, str: &str) {
        assert_eq!(
            set,
            &str.parse().expect("set in roundtrip could not be parsed")
        );
        assert_eq!(set.to_string(), str);
    }

    /// Test [`Mset::into_set`].
    #[test]
    fn into_set() {
        let a: Mset = "{{}, {}, {{}}, {{}, {}}}".parse().unwrap();
        roundtrip(&a.into_set(), "{{}, {{}}}");

        let b: Mset = "{{}, {}, {{{}}, {}, {}}, {{}, {}, {}}}".parse().unwrap();
        roundtrip(&b.into_set(), "{{}, {{}}, {{}, {{}}}}");
    }

    /// Test [`Set::empty`].
    #[test]
    fn empty() {
        roundtrip(&Set::empty(), "{}");
    }

    /// Test [`Set::singleton`].
    #[test]
    fn singleton() {
        let set = Set::empty().singleton();
        roundtrip(&set, "{{}}");
        let set = set.singleton();
        roundtrip(&set, "{{{}}}");
    }

    /// Test [`Set::pair`].
    #[test]
    fn pair() {
        let a = Set::empty();
        let b = Set::empty().singleton();
        let pair = a.clone().pair(b);
        roundtrip(&pair, "{{}, {{}}}");
        let pair = a.clone().pair(a);
        roundtrip(&pair, "{{}}");
    }

    /// Test [`Set::nat`].
    #[test]
    fn nat() {
        const NATS: [&str; 4] = ["{}", "{{}}", "{{}, {{}}}", "{{}, {{}}, {{}, {{}}}}"];
        for (n, nat) in NATS.iter().enumerate() {
            roundtrip(&Set::nat(n), nat);
        }
    }

    /*
    /// Test [`Set::union`].
    #[test]
    fn union() {
        let a: Mset = "{{}, {}, {{}}, {{}, {}}}".parse().unwrap();
        let b: Mset = "{{}, {}, {{{}}}, {{}, {}, {}}}".parse().unwrap();

        assert_eq!(&a, &a.clone().union(Mset::empty()));
        roundtrip(
            &a.union(b),
            "{{}, {}, {}, {}, {{}}, {{}, {}}, {{}, {}, {}}, {{{}}}}",
        );
    }

    /// Test [`Mset::inter`].
    #[test]
    fn inter() {
        let a: Mset = "{{}, {}, {{}}, {{{}}} {{}, {}}}".parse().unwrap();
        let b: Mset = "{{}, {}, {{{}}}, {{}, {}, {}}}".parse().unwrap();

        assert_eq!(&Mset::empty(), &a.clone().inter(Mset::empty()));
        roundtrip(&a.inter(b), "{{}, {}, {{{}}}}");
    }
    */
}
*/
