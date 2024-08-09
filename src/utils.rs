//! Utility types and algorithms for working with sets.
//!
//! Taking most of the space within the file is [`Levels`], a bespoke data structure responsible for
//! most of the clever algorithms for basic set manipulation.

use crate::{prelude::*, reuse_vec};
use std::ops::Range;

/// Assigns an increasing index to a `key` added to a tree, or returns the existing index.
pub(crate) fn btree_index<K: Ord>(tree: &mut BTreeMap<K, usize>, key: K) -> usize {
    let len = tree.len();
    match tree.entry(key) {
        Entry::Vacant(entry) => {
            entry.insert(len);
            len
        }
        Entry::Occupied(entry) => *entry.get(),
    }
}

/// A vector of vectors, flattened into a single vector plus a vector of indices. This saves on
/// allocations at the expense of not being able to add new elements to the vectors we've already
/// added.
///
/// We refer to each individual vector as a "level" of our nested vector.
///
/// ## Invariants
///
/// These invariants should hold for any [`NestVec`]. **Unsafe code can perform optimizations
/// contingent on these.**
///
/// - The elements of `indices` form an increasing sequence.
/// - All elements in `indices` are smaller than the length of `data`.
#[derive(Clone, Debug)]
pub struct NestVec<T> {
    /// The i-th element of the array represents the start point for the i-th level in the data
    /// array.
    indices: Vec<usize>,
    /// Stores all the data for all levels.
    data: Vec<T>,
}

// -------------------- Basic methods -------------------- //

impl<T> Default for NestVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> FromIterator<T> for NestVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            indices: vec![0],
            data: iter.into_iter().collect(),
        }
    }
}

impl<T> NestVec<T> {
    /// Initializes an empty [`NestVec`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Creates a new level holding the data from the specified iterator.
    pub fn push_iter<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.indices.push(self.data.len());
        self.data.extend(iter);
    }

    /// Creates a new level holding the specified value.
    pub fn push(&mut self, value: T) {
        self.push_iter([value]);
    }

    /// Creates a new [`NestVec`] with a single level holding the specified value.
    pub fn from_value(value: T) -> Self {
        Self::from_iter([value])
    }

    /// The number of levels stored.
    #[must_use]
    pub fn level_len(&self) -> usize {
        self.indices.len()
    }

    /// The total amount of data stored within all levels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the nested vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the range within the slice corresponding to some level.
    #[must_use]
    pub fn get_range(&self, level: usize) -> Option<Range<usize>> {
        let start = *self.indices.get(level)?;
        let end = self.indices.get(level + 1).copied().unwrap_or(self.len());
        Some(start..end)
    }

    /// Gets the slice corresponding to a given level.
    ///
    /// Returns the empty slice for any missing levels.
    #[must_use]
    pub fn get(&self, level: usize) -> &[T] {
        self.get_range(level)
            // Safety: our ranges are always valid for indexing.
            .map(|range| unsafe { self.data.get_unchecked(range) })
            .unwrap_or_default()
    }

    /// Gets the mutable slice corresponding to a given level.
    ///
    /// Returns the empty slice for any missing levels.
    pub fn get_mut(&mut self, level: usize) -> &mut [T] {
        self.get_range(level)
            // Safety: our ranges are always valid for indexing.
            .map(|range| unsafe { self.data.get_unchecked_mut(range) })
            .unwrap_or_default()
    }

    /// Builds the next level from the last.
    ///
    /// If the built level would be empty, does nothing and returns `false` instead.
    ///
    /// # Arguments
    ///
    /// - `extend`: a function extending an array based on some value `T`.
    /// - `buf`: a buffer for calculations.
    pub fn next_level_gen<F: FnMut(&mut Vec<T>, T)>(
        &mut self,
        mut extend: F,
        buf: &mut Vec<T>,
    ) -> bool
    where
        T: Copy,
    {
        // Get start and end of last level.
        let start;
        if let Some(s) = self.indices.last() {
            start = *s;
        } else {
            // Nothing to be built.
            return false;
        }
        let end = self.data.len();

        // Adds elements of each set in the last level.
        //
        // We write them into an auxiliary buffer first, as the reference to `set` might otherwise
        // be invalidated by an array resize.
        for i in start..end {
            // Safety: `i < end ≤ self.data.len()`
            let set = unsafe { *self.data.get_unchecked(i) };
            extend(buf, set);
            self.data.append(buf);
        }

        // Return whether the level is not empty.
        let cont = self.data.len() != end;
        if cont {
            self.indices.push(end);
        }
        cont
    }
}

impl<T> NestVec<&T> {
    /// Clears a nested vector and allows it to be reused for another lifetime.
    #[must_use]
    pub fn reuse<'a>(self) -> NestVec<&'a T> {
        let mut indices = self.indices;
        indices.clear();

        NestVec {
            indices,
            data: crate::reuse_vec(self.data),
        }
    }
}

// -------------------- Traits -------------------- //

impl<T: Display> Display for NestVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "[")?;

        let mut fst = false;
        for level in self.iter() {
            write!(f, "{}", if fst { "[" } else { "; [" })?;
            fst = false;

            let mut iter = level.iter();
            if let Some(next) = iter.next() {
                write!(f, "{next}")?;
                for next in iter {
                    write!(f, ",  {next}")?;
                }
            }
            write!(f, "]")?;
        }

        write!(f, "]")
    }
}

/// Shorthand for the traits our iterators implement.
macro_rules! traits {
    ($t: ty) => { impl DoubleEndedIterator<Item = $t> + ExactSizeIterator + '_ };
}

impl<T> NestVec<T> {
    /// Iterates over all levels.
    #[must_use]
    pub fn iter(&self) -> traits!(&[T]) {
        (0..self.level_len()).map(|r| self.get(r))
    }

    /// Mutably iterates over all levels.
    pub fn iter_mut(&mut self) -> traits!(&mut [T]) {
        let indices = &self.indices;
        let len = self.data.len();
        let ptr = self.data.as_mut_ptr();

        // Safety: these slices are all disjoint.
        indices.iter().enumerate().map(move |(r, start)| unsafe {
            let end = indices.get(r + 1).copied().unwrap_or(len);
            slice::from_raw_parts_mut(ptr.add(*start), end.unchecked_sub(*start))
        })
    }
}

/// An element that can be used within a [`Levels`]. This must be either `&Mset` or `*mut Mset`.
///
/// We don't implement this trait for the corresponding references to `Set` in the interest of
/// avoiding code duplication. We can't implement it for `&mut Mset`, as the [`Levels`] structure
/// would cause mutable aliasing.
pub trait SetPtr: Copy + crate::Seal {
    /// Reads the pointer and retrieves the cardinality of the [`Mset`].
    ///
    /// ## Safety
    ///
    /// In order to call this function, you must ensure the pointer can be dereferenced. Note that
    /// this is guaranteed for `&Mset`.
    unsafe fn read_card(self) -> usize;

    /// Extends a vector with the children of this item.
    ///
    /// ## Safety
    ///
    /// In order to call this function, you must ensure the pointer can be dereferenced. Note that
    /// this is guaranteed for `&Mset`.
    unsafe fn extend(vec: &mut Vec<Self>, set: Self);
}

impl crate::Seal for &Mset {}
impl SetPtr for &Mset {
    unsafe fn read_card(self) -> usize {
        self.card()
    }

    unsafe fn extend(vec: &mut Vec<Self>, set: Self) {
        vec.extend(set);
    }
}

impl crate::Seal for *mut Mset {}
impl SetPtr for *mut Mset {
    unsafe fn read_card(self) -> usize {
        (*self).card()
    }

    unsafe fn extend(vec: &mut Vec<Self>, set: Self) {
        vec.extend((*set).iter_mut().map(std::ptr::from_mut));
    }
}

// -------------------- Levels -------------------- //

/// A data structure storing references to all sets recursively contained within one or more
/// [`Mset`].
///
/// Many algorithms on sets, like [`Mset::eq`] or [`Mset::contains`], build a [`Levels`] structure
/// in their implementation.
///
/// ## Invariants
///
/// These invariants should hold for any [`Levels`]. **Unsafe code can perform optimizations
/// contingent on these.**
///
/// - None of the levels are empty.
///
/// At initialization, it should also satisfy:
///
/// - All pointers are valid to dereference.
/// - The sums of the cardinalities in one level equal the length of the next.
///
/// In order to work with `Levels<*mut Mset>`, you'll have to do so starting from the last level and
/// moving upwards. Otherwise, the pointers can get invalidated.
#[derive(Clone)]
#[repr(transparent)]
pub struct Levels<T: SetPtr>(NestVec<T>);

impl<T: SetPtr> NestVec<T> {
    /// Builds the next level from the last.
    ///
    /// If the built level would be empty, does nothing and returns false instead.
    ///
    /// ## Safety
    ///
    /// The pointers at the last level must be dereferenceable.
    pub unsafe fn next_level(&mut self, buf: &mut Vec<T>) -> bool {
        self.next_level_gen(|vec, set| T::extend(vec, set), buf)
    }

    /// Transmutes `&Self` into `&Levels<T>`.
    ///
    /// ## Safety
    ///
    /// You must guarantee the type invariants for [`Levels`].
    #[must_use]
    pub unsafe fn as_levels(&self) -> &Levels<T> {
        &*std::ptr::from_ref(self).cast()
    }

    /// Initializes a [`Levels`] from a nested vector, whose missing levels are built.
    ///
    /// ## Safety
    ///
    /// The nested vector must satisfy the invariants of the type, with the only exception that the
    /// cardinalities of the last level can add up to anything.
    #[must_use]
    pub unsafe fn build(mut self) -> Levels<T> {
        let mut buf = Vec::new();
        while self.next_level(&mut buf) {}
        Levels(self)
    }
}

impl<T: SetPtr> Levels<T> {
    /// Returns a reference to the inner nested vector.
    #[must_use]
    pub fn nest_vec(&self) -> &NestVec<T> {
        &self.0
    }

    /// Returns the inner nested vector.
    #[must_use]
    pub fn into_nest_vec(self) -> NestVec<T> {
        self.0
    }

    /// Initializes a [`Levels`] from an iterator for the first level.
    ///
    /// ## Safety
    ///
    /// The pointers returned by the iterator must be dereferenceable.
    pub unsafe fn new_iter_gen<I: IntoIterator<Item = T>>(iter: I) -> Self {
        NestVec::from_iter(iter).build()
    }

    /// Initializes a [`Levels`] from an entry for the first level.
    ///
    /// ## Safety
    ///
    /// The pointer must be dereferenceable.
    #[must_use]
    pub unsafe fn new_gen(set: T) -> Self {
        Self::new_iter_gen([set])
    }
}

impl<'a> Levels<&'a Mset> {
    /// Initializes a [`Levels`] from an iterator for the first level.
    pub fn new_iter<I: IntoIterator<Item = &'a Mset>>(iter: I) -> Self {
        // Safety: pointers from a reference are dereferenceable.
        unsafe { Self::new_iter_gen(iter) }
    }

    /// Initializes a [`Levels`] from an entry for the first level.
    #[must_use]
    pub fn new(set: &'a Mset) -> Self {
        // Safety: pointers from a reference are dereferenceable.
        unsafe { Self::new_gen(set) }
    }
}

impl Levels<*mut Mset> {
    /// Initializes a [`Levels`] from an iterator for the first level.
    pub fn new_iter_mut<'a, I: IntoIterator<Item = &'a mut Mset>>(iter: I) -> Self {
        // Safety: pointers from a reference are dereferenceable.
        unsafe { Self::new_iter_gen(iter.into_iter().map(ptr::from_mut)) }
    }

    /// Initializes a [`Levels`] from an entry for the first level.
    pub fn new_mut(set: &mut Mset) -> Self {
        // Safety: pointers from a reference are dereferenceable.
        unsafe { Self::new_gen(ptr::from_mut(set)) }
    }
}

impl Mset {
    /// Initializes two [`Levels`] simultaneously. Calls a function on every pair of levels built,
    /// which determines whether execution is halted early.
    pub fn both_levels<'a, F: FnMut(&[&'a Self], &[&'a Self]) -> bool>(
        &'a self,
        other: &'a Self,
        mut cb: F,
    ) -> Option<(Levels<&'a Self>, Levels<&'a Self>)> {
        let mut fst = NestVec::from_value(self);
        let mut snd = NestVec::from_value(other);
        let mut cont_fst = true;
        let mut cont_snd = true;
        let mut level = 1;
        let mut buf = Vec::new();

        loop {
            // Step execution.
            // Safety: references can always be dereferenced.
            unsafe {
                if cont_fst {
                    cont_fst = fst.next_level(&mut buf);
                }
                if cont_snd {
                    cont_snd = snd.next_level(&mut buf);
                }
            }

            // Check if finished.
            if !cont_fst && !cont_snd {
                return Some((Levels(fst), Levels(snd)));
            }

            // Condition fail.
            if !cb(fst.get(level), snd.get(level)) {
                return None;
            }
            level += 1;
        }
    }

    /// Initializes two [`Levels`] in the procedure to check set equality. Returns `None` if we can
    /// prove that they're not equal before the structures are fully built.
    #[must_use]
    pub fn eq_levels<'a>(
        self: &'a Mset,
        other: &'a Mset,
    ) -> Option<(Levels<&'a Self>, Levels<&'a Self>)> {
        self.both_levels(other, |fst, snd| fst.len() == snd.len())
    }

    /// Initializes two [`Levels`] in the procedure to check subsets. Returns `None` if we can prove
    /// that they're not subsets before the structures are fully built.
    #[must_use]
    pub fn le_levels<'a>(
        self: &'a Mset,
        other: &'a Mset,
    ) -> Option<(Levels<&'a Self>, Levels<&'a Self>)> {
        self.both_levels(other, |fst, snd| fst.len() <= snd.len())
    }
}

impl<T: SetPtr> Levels<T> {
    /// For each set in a level within [`Levels`], finds the range for its children in the next
    /// level.
    ///
    /// ## Safety
    ///
    /// The pointers within this level must be dereferenceable.
    #[must_use]
    pub unsafe fn children(&self, level: usize) -> traits!(Range<usize>) {
        let mut start = 0;
        self.0.get(level).iter().map(move |set| {
            let end = start + set.read_card();
            let range = start..end;
            start = end;
            range
        })
    }

    /// For each set in a level within [`Levels`], finds the slice representing its children in the
    /// next level, then uses it to index a separate slice.
    ///
    /// ## Safety
    ///
    /// The pointers within this level must be dereferenceable. Moreover, the indexed slice must
    /// have at least as many elements as the next level.
    #[must_use]
    pub unsafe fn children_slice<'a, U>(
        &'a self,
        level: usize,
        slice: &'a [U],
    ) -> traits!(&'a [U]) {
        self.children(level)
            .map(move |range| slice.get_unchecked(range))
    }

    /// For each set in a level within [`Levels`], finds the slice representing its children in the
    /// next level, then uses it to mutably index a separate slice.
    ///
    /// ## Safety
    ///
    /// The pointers within this level must be dereferenceable. Moreover, the indexed slice must
    /// have at least as many elements as the next level.
    #[must_use]
    pub unsafe fn children_mut_slice<'a, U>(
        &'a self,
        level: usize,
        slice: &'a mut [U],
    ) -> traits!(&'a mut [U]) {
        let next = slice.as_mut_ptr();
        // Safety: all these slices are disjoint.
        self.children(level)
            .map(move |range| slice::from_raw_parts_mut(next.add(range.start), range.len()))
    }
}

// -------------------- AHU algorithm -------------------- //

impl<T: SetPtr> Levels<T> {
    /// Performs one step of the modified AHU algorithm.
    ///
    /// Transforms some set of values assigned to the children of a level, into values for the
    /// level, via a specified function. The algorithm stops and returns false if `None` is returned
    /// by said function.
    ///
    /// ## Arguments
    ///
    /// - `level`: a level within [`Levels`].
    /// - `cur`: a buffer to write the output.
    /// - `next`: the values associated to the next level.
    /// - `child_fn`: the function mapping the slice of children into the assigned value.
    ///
    /// ## Safety
    ///
    /// The pointers within this level must be dereferenceable. Moreover, `next` must contain at
    /// least as many values as the next level.
    pub unsafe fn step_ahu<U, V, F: FnMut(&mut [U], T) -> Option<V>>(
        &self,
        level: usize,
        cur: &mut Vec<V>,
        next: &mut [U],
        mut child_fn: F,
    ) -> bool {
        cur.clear();
        let lev = self.0.get(level);
        // Safety: the indexed slice contains as many elements as required due to the invariant on
        // `self`.
        for (i, slice) in self.children_mut_slice(level, next).enumerate() {
            if let Some(idx) = child_fn(slice, *lev.get_unchecked(i)) {
                cur.push(idx);
            } else {
                return false;
            }
        }

        true
    }

    /// Performs the modified AHU algorithm up to the specified level.
    ///
    /// Transforms some set of values assigned to the children of a level, into values for the
    /// level, via a specified function. The algorithm stops and returns false if `None` is returned
    /// by said function.
    ///
    /// ## Arguments
    ///
    /// - `level`: a level within [`Levels`].
    /// - `child_fn`: the function mapping the slice of children into the assigned value.
    /// - `sets`: an optional auxiliary structure to use within `child_fn`.
    /// - `level_fn`: a function resetting `sets` after each level.
    ///
    /// ## Safety
    ///
    /// The pointers within all levels must be dereferenceable.
    pub unsafe fn mod_ahu_gen<
        U,
        V,
        F: FnMut(&mut V, &mut [U], T) -> Option<U>,
        G: FnMut(&mut V),
    >(
        &self,
        level: usize,
        mut sets: V,
        mut child_fn: F,
        mut level_fn: G,
    ) -> Option<Vec<U>> {
        let mut cur = Vec::new();
        let mut next = Vec::new();

        for level in (level..self.0.level_len()).rev() {
            if !self.step_ahu(level, &mut cur, &mut next, |i, j| child_fn(&mut sets, i, j)) {
                return None;
            }

            level_fn(&mut sets);
            mem::swap(&mut cur, &mut next);
        }

        Some(next)
    }
}

impl<'a> Levels<&'a Mset> {
    /// Performs the modified AHU algorithm up to the specified level.
    ///
    /// Transforms some set of values assigned to the children of a level, into values for the
    /// level, via a specified function. The algorithm stops and returns false if `None` is returned
    /// by said function.
    ///
    /// See [`Self::mod_ahu_gen`].
    pub fn mod_ahu<U, V, F: FnMut(&mut V, &mut [U], &Mset) -> Option<U>, G: FnMut(&mut V)>(
        &self,
        level: usize,
        sets: V,
        child_fn: F,
        level_fn: G,
    ) -> Option<Vec<U>> {
        // Safety: pointers from a reference are dereferenceable.
        unsafe { self.mod_ahu_gen(level, sets, child_fn, level_fn) }
    }

    /// The simplest and most common instantiation of [`Self::mod_ahu`], where we simply find unique
    /// labels for the sets at a given level.
    pub fn ahu(&self, level: usize) -> Vec<usize> {
        let ahu = self.mod_ahu(
            level,
            BTreeMap::new(),
            |sets, slice, _| {
                slice.sort_unstable();
                let children: SmallVec<_> = slice.iter().copied().collect();
                Some(btree_index(sets, children))
            },
            BTreeMap::clear,
        );

        // Safety: `Some(x) != None`.
        unsafe { ahu.unwrap_unchecked() }
    }
}

/// The [Aho–Hopcroft–Ullman](https://www.baeldung.com/cs/isomorphic-trees) (AHU) encoding for an
/// [`Mset`]. It is unique up to multiset equality.
///
/// Conceptually, this amounts to hereditarily lexicographically ordered roster notation. In fact,
/// the [`Display`] implementation for [`Mset`] constructs an [`Ahu`] first.
///
/// ## Modified AHU algorithm
///
/// A practical issue with the AHU encoding is that after the first few levels, it becomes expensive
/// to store and compare all of the partial encodings. As such, instead of computing the full AHU
/// encoding, we often opt for a modified encoding, where at each step, each unique multiset is
/// assigned a single integer instead of the full string. This "modified" AHU encoding does not
/// determine multisets uniquely, but it can uniquely determine multisets within a single multiset.
///
/// See [`Levels::mod_ahu`] for an implementation.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, IntoIterator)]
pub struct Ahu(#[into_iterator(owned, ref)] BitVec);

impl Ahu {
    /// The empty encoding.
    #[must_use]
    pub const fn empty() -> Self {
        Self(BitVec::EMPTY)
    }

    /// Finds the [`Ahu`] encodings for an iterator over multisets.
    pub fn new_iter<'a, I: IntoIterator<Item = &'a Mset>>(iter: I) -> Vec<Self> {
        /// Avoid code duplication.
        fn new_iter_levels(levels: &Levels<&Mset>) -> Vec<Ahu> {
            levels
                .mod_ahu(
                    0,
                    (),
                    |(), slice, _| {
                        // Reuse buffer. Add enclosing parentheses.
                        slice.sort_unstable();
                        let mut iter = slice.iter_mut();
                        let fst;
                        if let Some(f) = iter.next() {
                            fst = f;
                        } else {
                            return Some(Ahu::empty());
                        }
                        let mut buf = mem::take(fst);

                        // Closing parenthesis.
                        buf.0.push(false);
                        buf.0.push(false);
                        buf.0.shift_right(1);
                        buf.0.set(0, true);
                        // Opening parenthesis.

                        for set in iter {
                            buf.0.push(true);
                            buf.0.append(&mut set.0);
                            buf.0.push(false);
                        }
                        Some(buf)
                    },
                    |()| {},
                )
                .unwrap()
        }

        new_iter_levels(&Levels::new_iter(iter))
    }

    /// Finds the [`Ahu`] encoding for a multiset.
    #[must_use]
    pub fn new(set: &Mset) -> Self {
        // Safety: the top level of our Levels has a root node.
        unsafe { Self::new_iter([set]).pop().unwrap_unchecked() }
    }
}

impl Debug for Ahu {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_char('(')?;
        for b in self {
            f.write_char(if *b { '(' } else { ')' })?;
        }
        f.write_char(')')
    }
}

impl Display for Ahu {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_char('{')?;
        let mut last = true;
        for b in self {
            if *b {
                if !last {
                    f.write_str(", ")?;
                }
                f.write_char('{')
            } else {
                f.write_char('}')
            }?;

            last = *b;
        }

        f.write_char('}')
    }
}

// -------------------- Comparison -------------------- //

impl<'a> Levels<&'a Mset> {
    /// Returns whether `self` is a subset of `other`, meaning it contains each set at least as many
    /// times.
    ///
    /// ## Precalculations
    ///
    /// It can save a lot of time to first perform basic checks as the levels are built. For
    /// instance, if some level of `self` has more elements than the corresponding level of `other`,
    /// it can't be a subset, and we don't need to build the rest of the levels. Similarly, if all
    /// levels have the same number of elements, the subset relation actually implies equality.
    #[must_use]
    pub fn subset(&self, other: &Self) -> bool {
        // This check is often redundant, but it guarantees correctness of the function in all
        // cases, and it's not like it's a bottleneck.
        if self.0.level_len() > other.0.level_len() {
            return false;
        }

        let mut cur = Vec::new();
        let mut fst_next = Vec::new();
        let mut snd_next = Vec::new();

        let mut sets = BTreeMap::new();
        for level in (1..other.0.level_len()).rev() {
            sets.clear();

            // Safety: the invariant is handled just the same as in `Self::mod_ahu`, just with two
            // sets at once.
            unsafe {
                // Processs second set.
                other.step_ahu(level, &mut cur, &mut snd_next, |slice, _| {
                    let mut children: SmallVec<_> = slice.iter().copied().collect();
                    children.sort_unstable();

                    // Increment set count.
                    let len = sets.len();
                    match sets.entry(children) {
                        Entry::Vacant(entry) => {
                            entry.insert((len, 0));
                            Some(len)
                        }
                        Entry::Occupied(mut entry) => {
                            let (idx, num) = entry.get_mut();
                            *num += 1;
                            Some(*idx)
                        }
                    }
                });
                mem::swap(&mut cur, &mut snd_next);

                // Process first set.
                let res = self.step_ahu(level, &mut cur, &mut fst_next, |slice, _| {
                    let mut children: SmallVec<_> = slice.iter().copied().collect();
                    children.sort_unstable();

                    // Decrement set count. Return if this reaches a negative.
                    match sets.entry(children) {
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
                    }
                });
                if !res {
                    return false;
                }
                mem::swap(&mut cur, &mut fst_next);
            }
        }

        true
    }
}

/// An auxiliary structure to efficiently compare a given set to multiple others.
///
/// In order to test equality between two sets, we build a [`Levels`] structure for each. In the
/// case where we need to compare multiple sets with a single one, this avoids recomputation of the
/// first [`Levels`] structure, and re-allocation of the buffer for the second one.
///
/// If you only need to compare two sets, use [`Mset::eq_levels`] or [`Mset::le_levels`] instead, as
/// they avoid allocating the entire [`Levels`] structure if the comparison can be proved false
/// before that.
pub struct Compare<'a> {
    /// The set to compare others against.
    set: Levels<&'a Mset>,
    /// A structure in which we store the [`Levels`] for the other sets. The allocation gets reused.
    ///
    /// We can't store `Levels<&'a Mset>` directly, as having an empty allocation would break the
    /// type invariants.
    other: NestVec<&'a Mset>,
    /// A buffer for calculations we reuse.
    buf: Vec<&'a Mset>,
}

impl<'a> Compare<'a> {
    /// Initializes a [`Compare`] structure for the given set.
    #[must_use]
    pub fn new(set: &'a Mset) -> Self {
        Self {
            set: Levels::new(set),
            other: NestVec::new(),
            buf: Vec::new(),
        }
    }

    /// Returns the cardinality of the stored set.
    #[must_use]
    pub fn card(&self) -> usize {
        self.set.0.get(1).len()
    }

    /// Combines the functions [`Self::eq`], [`Self::le`], and [`Self::ge`].
    fn cmp_with<F: FnMut(usize, usize) -> bool>(&mut self, other: &Mset, mut cmp: F) -> bool {
        // We could optimize this by not clearing the buffers twice. They should already be empty
        // whenever this function is called.
        let mut levels = mem::take(&mut self.other).reuse();
        let mut buf = reuse_vec(mem::take(&mut self.buf));
        levels.push(other);

        // Safety: by building our levels manually, we guarantee the type invariants.
        let res = unsafe {
            let mut idx = 1;
            while levels.next_level(&mut buf) {
                let level = self.set.0.get(idx);
                if level.is_empty() || !cmp(level.len(), levels.get(idx).len()) {
                    self.other = levels.reuse();
                    return false;
                }
                idx += 1;
            }

            self.set.subset(levels.as_levels())
        };

        self.other = levels.reuse();
        self.buf = reuse_vec(buf);
        res
    }

    /// Tests equality with another set.
    pub fn eq(&mut self, other: &Mset) -> bool {
        self.cmp_with(other, |x, y| x == y)
    }
    /// Tests inequality with another set.
    pub fn ne(&mut self, other: &Mset) -> bool {
        !self.eq(other)
    }

    /// Tests subset with another set.
    pub fn le(&mut self, other: &Mset) -> bool {
        self.cmp_with(other, |x, y| x <= y)
    }
    /// Tests superset with another set.
    pub fn ge(&mut self, other: &Mset) -> bool {
        self.cmp_with(other, |x, y| x >= y)
    }

    /// Tests strict subset with another set.
    pub fn lt(&mut self, other: &Mset) -> bool {
        self.card() < other.card() && self.le(other)
    }
    /// Tests strict superset with another set.
    pub fn gt(&mut self, other: &Mset) -> bool {
        self.card() > other.card() && self.ge(other)
    }

    /// Compares two sets.
    pub fn partial_cmp(&mut self, other: &Mset) -> Option<Ordering> {
        let cmp = self.card().cmp(&other.card());
        let test = match cmp {
            Ordering::Equal => self.eq(other),
            Ordering::Less => self.le(other),
            Ordering::Greater => self.ge(other),
        };

        if test {
            Some(cmp)
        } else {
            None
        }
    }
}
