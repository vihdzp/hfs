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
        unsafe { crate::transmute_vec(set.0 .0) }
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
        let el = std::mem::take(set.get_unchecked_mut(*i));
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

        levels.mod_ahu_gen(
            1,
            BTreeMap::new(),
            // Safety: Since we're modifying sets from bottom to top, we can ensure our pointers are
            // still valid.
            |s| unsafe { &*(s.cast_const()) }.card(),
            |sets, slice, &set| {
                // Deduplicate the set.
                unsafe {
                    dedup_by(&mut (*set).0, slice, &mut buf, &mut buf_pairs);
                };

                let children: SmallVec<_> = buf_pairs.iter().map(|(_, k)| *k).collect();
                Some(btree_index(sets, children))
            },
            BTreeMap::clear,
        );

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
    pub unsafe fn cast_vec(vec: Vec<Self>) -> Vec<Set> {
        crate::transmute_vec(vec)
    }
}

impl Set {
    /// Deduplicate a vector of [`Set`].
    ///
    /// This is optimized compared to calling [`Mset::into_set`] on the vector, as we don't need to
    /// deduplicate any levels further down.
    pub fn dedup(mut vec: Vec<Self>) -> Self {
        let keys = Levels::init_iter(vec.iter().map(AsRef::as_ref))
            .fill()
            .ahu(2);

        unsafe { dedup_by(&mut vec, &keys, &mut Vec::new(), &mut Vec::new()) };
        Self(Mset(Self::cast_vec(vec)))
    }

    /// Converts `Vec<Set>` into `Vec<Mset>`.
    pub fn cast_vec(vec: Vec<Self>) -> Vec<Mset> {
        unsafe { crate::transmute_vec(vec) }
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

impl<'a> Iterator for Cast<std::slice::Iter<'a, Mset>> {
    type Item = &'a Set;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|s| unsafe { s.as_set_unchecked() })
    }
}

impl<'a> Iterator for Cast<std::slice::IterMut<'a, Mset>> {
    type Item = &'a mut Set;

    fn next(&mut self) -> Option<Self::Item> {
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

// -------------------- SetTrait -------------------- //

impl crate::Seal for Set {}

impl SetTrait for Set {
    // -------------------- Basic methods -------------------- //

    fn as_slice(&self) -> &[Self] {
        let slice = self.0.as_slice();
        unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
    }

    fn as_vec(&self) -> &Vec<Mset> {
        self.0.as_vec()
    }

    fn clear(&mut self) {
        self.0.clear();
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
            .select_mut(|set| pred(unsafe { set.as_set_unchecked() }));
    }

    fn union(self, other: Self) -> Self {
        Self::dedup(unsafe { Mset::cast_vec((self.0.union(other.0)).0) })
    }

    fn union_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        Self::dedup(unsafe { Mset::cast_vec(Mset::union_iter(iter.into_iter().map(Into::into)).0) })
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
                // Very cheeky and probably unhelpful use of unsafe code.
                if sets.insert(children, len).is_some() {
                    unsafe { std::hint::unreachable_unchecked() }
                }
                len
            },
        )
    }

    fn disjoint(&self, other: &Self) -> bool {
        Self::disjoint_pairwise([self, other])
    }

    fn disjoint_pairwise<'a, I: IntoIterator<Item = &'a Self>>(iter: I) -> bool {
        // Empty sets are disjoint.
        let levels = Levels::init_iter(iter.into_iter().map(AsRef::as_ref)).fill();
        let elements;
        if let Some(el) = levels.get(1) {
            elements = el;
        } else {
            return true;
        }

        let mut cur = Vec::new();
        let mut next: Vec<usize> = Vec::new();
        let mut sets = BTreeMap::new();

        // Compute AHU encodings for all but the elements of the union.
        for level in levels.iter().skip(2).rev() {
            sets.clear();
            unsafe {
                Levels::step_ahu(level, &mut cur, &mut next, |slice, _| {
                    slice.sort_unstable();
                    let children = slice.iter().copied().collect::<SmallVec<_>>();
                    Some(btree_index(&mut sets, children))
                })
            };
        }

        // Compute the encodings for the union. Return whether we find anything twice.
        let mut dummy: Vec<()> = Vec::new();
        sets.clear();
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
    pub unsafe fn as_slice_mut(&mut self) -> &mut [Self] {
        let slice = self.0.as_slice_mut();
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
    }

    /// A mutable reference to the inner vector.
    ///
    /// ## Safety
    ///
    /// You must preserve the type invariants for [`Set`]. In particular, you can't make two
    /// elements equal.
    pub unsafe fn as_vec_mut(&mut self) -> &mut Vec<Mset> {
        &mut self.0 .0
    }

    /// Mutably iterate over the elements of the set.
    ///
    /// ## Safety
    ///
    /// You must preserve the type invariants for [`Set`]. In particular, you can't make two
    /// elements equal.
    pub unsafe fn iter_mut(&mut self) -> std::slice::IterMut<Self> {
        self.as_slice_mut().iter_mut()
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
    /// Set union x ∪ y.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        (self.0.union(other.0)).into_set()
    }

    /// Set union ∪x.
    #[must_use]
    pub fn big_union(self) -> Self {
        self.0.big_union().into_set()
    }

    /// Set union ∪x.
    pub fn big_union_vec(vec: Vec<Self>) -> Self {
        let union: Vec<Mset> = vec.into_iter().flatten().map(Into::into).collect();
        Mset(union).into_set()
    }

    /// Set intersection x ∩ y.
    ///
    /// This is a modified version of [`Mset::inter`].
    #[must_use]
    pub fn inter(mut self, mut other: Self) -> Self {
        let idx = self.card();
        if idx == 0 || other.is_empty() {
            return Self::empty();
        }

        let levels = Levels::init_iter([self.mset(), other.mset()]).fill();
        let elements = unsafe { levels.get(1).unwrap_unchecked() };

        // We store the indices of the sets in the intersection.
        let mod_ahu = levels.test_mod_ahu(2);
        let mut next = mod_ahu.next;
        let mut indices = mod_ahu.buffer;

        // Each entry stores the index where it's found within the first set.
        let mut sets = BTreeMap::new();
        for (i, slice) in unsafe { Levels::child_iter_mut(elements, &mut next) }.enumerate() {
            slice.sort_unstable();
            let children: SmallVec<_> = slice.iter().copied().collect();

            match sets.entry(children) {
                Entry::Vacant(entry) => {
                    if i < idx {
                        entry.insert(i);
                    }
                }
                Entry::Occupied(entry) => {
                    debug_assert!(
                        i >= idx,
                        "there can't be repeated elements within a single set"
                    );
                    indices.push(entry.remove());
                }
            }
        }

        other.clear();
        for i in indices {
            let set = std::mem::take(unsafe { self.as_slice_mut().get_unchecked_mut(i) });
            other.0.insert_mut(set.0);
        }

        other
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
    #[must_use]
    pub fn powerset(self) -> Self {
        Self(self.0.powerset())
    }

    /// The von Neumann rank of the set.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    /// The von Neumann set encoding for n.
    #[must_use]
    pub fn nat(n: usize) -> Self {
        Self(Mset::nat(n))
    }

    /// The Zermelo set encoding for n.
    #[must_use]
    pub fn zermelo(n: usize) -> Self {
        Self(Mset::zermelo(n))
    }

    /// The von Neumann hierarchy.
    #[must_use]
    pub fn neumann(n: usize) -> Self {
        Self(Mset::neumann(n))
    }

    /// Kuratowski pair (x, y).
    #[must_use]
    pub fn k_pair(self, other: Self) -> Self {
        self.clone().singleton().pair(self.pair(other))
    }

    /// Decomposes a Kuratowski pair.
    #[must_use]
    pub fn k_split(&self) -> Option<(&Self, &Self)> {
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
}

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
