//! Utility types and algorithms for working with sets.
//!
//! Taking most of the space within the file is [`Levels`], a bespoke data structure responsible for
//! most of the clever algorithms for basic set manipulation.

use crate::prelude::*;
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

/// Represents the multisets at each rank within one or more [`Mset`], and assigns some data to
/// each. Most complex algorithms for sets, like [`Mset::contains`] or [`Mset::subset`], are
/// actually implemented in terms of this type.
///
/// To save on allocations, we use a single vector and an "indexing" vector to get subslices of it,
/// but morally, this is a `Vec<Vec<T>>`. As a further optimization, many functions take in an
/// initial [`Levels`] to use as a buffer, which can be built from [`Self::init`] or reused with
/// [`Self::init_with`].
///
/// ## Invariants
///
/// These invariants should hold for any [`Levels`]. **Unsafe code performs optimizations contingent
/// on these.**
///
/// - Every [`Levels`] must have a root level, and no level can be empty.
/// - The elements of `indices` are a strictly increasing sequence, and they are all smaller than
///   the length of `data`.
#[derive(Clone, Debug)]
pub struct Levels<T> {
    /// The i-th element of the array represents the start point for the i-th level in the data
    /// array.
    indices: SmallVec<usize>,
    /// Stores all the data for all levels.
    data: Vec<T>,
}

// -------------------- Basic methods -------------------- //

impl<T> Levels<T> {
    /// Initializes an empty [`Levels`]. **This breaks type invariants**.
    ///
    /// ## Safety
    ///
    /// The only valid use for this is as a placeholder or as an initial buffer to be immediately
    /// filled.
    #[must_use]
    pub unsafe fn empty() -> Self {
        Self {
            indices: SmallVec::new(),
            data: Vec::new(),
        }
    }

    /// Initializes the first level from a set.
    pub fn init(set: T) -> Self {
        Self {
            indices: smallvec![0],
            data: vec![set],
        }
    }

    /// Initializes the first level from a set. Reuses the specified buffer.
    pub fn init_mut(&mut self, set: T) {
        self.indices.clear();
        self.indices.push(0);
        self.data.clear();
        self.data.push(set);
    }

    /// Initializes the first level from a set. Reuses the specified buffer.
    #[must_use]
    pub fn init_with(mut self, set: T) -> Self {
        self.init_mut(set);
        self
    }

    /// Initializes the first level from a list of sets.
    pub fn init_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            indices: smallvec![0],
            data: iter.into_iter().collect(),
        }
    }

    /// The number of levels stored.
    #[must_use]
    pub fn level_len(&self) -> usize {
        self.indices.len()
    }

    /// The rank of the set, one less than the number of levels.
    #[must_use]
    pub fn rank(&self) -> usize {
        unsafe { self.level_len().unchecked_sub(1) }
    }

    /// The total amount of data stored within all levels.
    #[allow(clippy::len_without_is_empty)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get the range within the slice corresponding to some level.
    #[must_use]
    pub fn get_range(&self, level: usize) -> Option<Range<usize>> {
        let start = *self.indices.get(level)?;
        let end = self.indices.get(level + 1).copied().unwrap_or(self.len());
        Some(start..end)
    }

    /// Gets the slice corresponding to a given level.
    #[must_use]
    pub fn get(&self, level: usize) -> Option<&[T]> {
        self.get_range(level)
            .map(|range| unsafe { self.data.get_unchecked(range) })
    }

    /// Gets the mutable slice corresponding to a given level.
    pub fn get_mut(&mut self, level: usize) -> Option<&mut [T]> {
        self.get_range(level)
            .map(|range| unsafe { self.data.get_unchecked_mut(range) })
    }

    /// Returns the last element in `indices`.
    #[must_use]
    pub fn last_idx(&self) -> usize {
        unsafe { *self.indices.last().unwrap_unchecked() }
    }

    /// Returns the last level.
    #[must_use]
    pub fn last(&self) -> &[T] {
        let idx = self.last_idx();
        unsafe { self.data.get_unchecked(idx..) }
    }

    /// Returns a mutable reference to the last level.
    pub fn last_mut(&mut self) -> &mut [T] {
        let idx = self.last_idx();
        unsafe { self.data.get_unchecked_mut(idx..) }
    }
}

/// Shorthand for the traits our iterators implement.
macro_rules! traits {
    ($t: ty, $l: lifetime) => { impl DoubleEndedIterator<Item = $t> + ExactSizeIterator + $l };
    ($t: ty) => { traits!($t, '_) };
}

// -------------------- Filling -------------------- //

impl<T> Levels<T> {
    /// Builds the next level from the last. Returns whether this level was nonempty.
    ///
    /// - `T`: pointer type to a set-like object
    /// - `extend`: a function extending an array with the children of a set `T`
    /// - `buf`: a buffer for calculations
    pub fn step_gen<F: FnMut(&mut Vec<T>, T)>(&mut self, mut extend: F, buf: &mut Vec<T>) -> bool
    where
        T: Copy,
    {
        // Gets the last level.
        let start = self.last_idx();
        let end = self.data.len();

        // Adds elements of each set in the last level.
        //
        // We write them into an auxiliary buffer first, as the reference to `set` might otherwise
        // be invalidated by an array resize.
        for i in start..end {
            let set = unsafe { *self.data.get_unchecked(i) };
            extend(buf, set);
            self.data.extend(&*buf);
            buf.clear();
        }

        // Return whether the level is not empty.
        let cont = self.data.len() != end;
        if cont {
            self.indices.push(end);
        }
        cont
    }

    /// Fills all the [`Levels`] until there are no more children.
    ///
    /// See [`Self::fill`] and [`Self::fill_mut`] for specific instantiations.
    ///
    /// - `T`: pointer type to a set-like object
    /// - `extend`: a function extending an array with the children of a set `T`
    #[must_use]
    pub fn fill_gen<F: FnMut(&mut Vec<T>, T)>(mut self, mut extend: F) -> Self
    where
        T: Copy,
    {
        let mut buf = Vec::new();
        while self.step_gen(&mut extend, &mut buf) {}
        self
    }
}

/// We don't implement Levels<&Set> to avoid code duplication.
impl<'a> Levels<&'a Mset> {
    /// Builds the next level from the last. Returns whether this level was nonempty.
    ///
    /// See [`Self::step_gen`].
    pub fn step(&mut self, buf: &mut Vec<&'a Mset>) -> bool {
        self.step_gen(Vec::extend, buf)
    }

    /// Fills levels for an [`Mset`].
    #[must_use]
    pub fn fill(self) -> Self {
        self.fill_gen(Vec::extend)
    }

    /// Initializes two [`Levels`] simultaneously. Calls a function on every pair of levels built,
    /// which determines whether execution is halted early.
    pub fn both<F: FnMut(&[&'a Mset], &[&'a Mset]) -> bool>(
        mut self,
        mut other: Self,
        mut cb: F,
    ) -> Option<(Self, Self)> {
        let mut cont_fst = false;
        let mut cont_snd = false;
        let mut level = 1;
        let mut buf = Vec::new();

        loop {
            // Step execution.
            if cont_fst {
                cont_fst = self.step(&mut buf);
            }
            if cont_snd {
                cont_snd = other.step(&mut buf);
            }

            // Check if finished.
            if !cb(
                self.get(level).unwrap_or(&[]),
                other.get(level).unwrap_or(&[]),
            ) {
                return None;
            }
            if !(cont_fst || cont_snd) {
                return Some((self, other));
            }
            level += 1;
        }
    }

    /// Initializes two [`Levels`] in the procedure to check set equality.
    pub(crate) fn eq_levels(fst: &'a Mset, snd: &'a Mset) -> Option<(Self, Self)> {
        Self::init(fst).both(Self::init(snd), |fst, snd| fst.len() == snd.len())
    }

    /// Initializes two [`Levels`] in the procedure to check subsets.
    pub(crate) fn le_levels(fst: &'a Mset, snd: &'a Mset) -> Option<(Self, Self)> {
        Self::init(fst).both(Self::init(snd), |fst, snd| fst.len() <= snd.len())
    }
}

impl Levels<*mut Mset> {
    /// Fills mutable levels for a [`Mset`]. Pointers are reqiured as each level mutably aliases the
    /// next.
    ///
    /// ## Safety
    ///
    /// This method is completely safe, but you must be careful dereferencing pointers. Modifying a
    /// set and trying to access its children will often result in an invalid dereference.
    #[must_use]
    pub fn fill_mut(self) -> Self {
        // The set is not mutated, so the pointers remain valid to dereference.
        self.fill_gen(|buf, set| {
            buf.extend(unsafe { &mut *set }.iter_mut().map(std::ptr::from_mut));
        })
    }
}

// -------------------- Iterators -------------------- //

impl<T> Levels<T> {
    /// Iterates over all levels.
    #[must_use]
    pub fn iter(&self) -> traits!(&[T]) {
        (0..self.level_len()).map(|r| unsafe { self.get(r).unwrap_unchecked() })
    }

    /// Mutably iterates over all levels.
    pub fn iter_mut(&mut self) -> traits!(&mut [T]) {
        // Safety: these slices are all disjoint.
        let indices = &self.indices;
        let len = self.data.len();
        let ptr = self.data.as_mut_ptr();
        indices.iter().enumerate().map(move |(r, start)| unsafe {
            let end = indices.get(r + 1).copied().unwrap_or(len);
            std::slice::from_raw_parts_mut(ptr.add(*start), end.unchecked_sub(*start))
        })
    }

    /// For each set in a level within [`Levels`], finds the range for its children in the next
    /// level. Allows for a custom cardinality function.
    ///
    /// See also [`Self::child_iter`].
    pub fn child_iter_gen<'a, F: FnMut(&T) -> usize + 'a>(
        level: &'a [T],
        mut card: F,
    ) -> traits!(Range<usize>, 'a) {
        let mut start = 0;
        level.iter().map(move |set| {
            let end = start + card(set);
            let range = start..end;
            start = end;
            range
        })
    }
}

impl<T: Display> Display for Levels<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // Number of padding digits needed.
        let mut rank = self.rank();
        let mut digits = 1;
        while rank >= 10 {
            rank /= 10;
            digits += 1;
        }

        for (i, level) in self.iter().enumerate() {
            let mut iter = level.iter();

            write!(f, "Level {i:>digits$}: {}", unsafe {
                iter.next().unwrap_unchecked()
            })?;

            for next in iter {
                write!(f, " | {next}")?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

impl<'a> Levels<&'a Mset> {
    /// For each set in a level within [`Levels`], finds the range for its children in the next
    /// level.
    ///
    /// The functions [`Self::child_iter`] and [`Self::child_iter_mut`] are often more convenient
    /// but are unsafe.
    #[must_use]
    pub fn child_iter_range(level: &'a [&'a Mset]) -> traits!(Range<usize>) {
        Self::child_iter_gen(level, |s| s.card())
    }

    /// For each set in a level within [`Levels`], finds the slice representing its children in the
    /// next level.
    ///
    /// ## Safety
    ///
    /// The sum of the cardinalities in `level` cannot exceed the length of `next`.
    #[must_use]
    pub fn child_iter<T>(level: &'a [&'a Mset], next: &'a [T]) -> traits!(&'a [T], 'a) {
        Self::child_iter_range(level).map(move |range| unsafe { next.get_unchecked(range) })
    }

    /// For each set in a level within [`Levels`], finds the mutable slice representing its children
    /// in the next level.
    ///
    /// ## Safety
    ///
    /// The sum of the cardinalities in `level` cannot exceed the length of `next`.
    pub unsafe fn child_iter_mut<T>(
        level: &'a [&'a Mset],
        next: &'a mut [T],
    ) -> traits!(&'a mut [T], 'a) {
        // Safety: all these slices are disjoint.
        let next = next.as_mut_ptr();
        Self::child_iter_range(level)
            .map(move |range| unsafe { std::slice::from_raw_parts_mut(next, range.len()) })
    }
}

// -------------------- AHU algorithm -------------------- //

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
/// See [`Levels::mod_ahu`] for an implementation. Note that variations of this are implemented all
/// over the place.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, IntoIterator)]
pub struct Ahu(#[into_iterator(owned, ref)] BitVec);

impl Ahu {
    /// The empty encoding.
    #[must_use]
    pub const fn empty() -> Self {
        Self(BitVec::EMPTY)
    }

    /// Finds the [`Ahu`] encoding for a multiset.
    #[must_use]
    pub fn new(set: &Mset) -> Self {
        let levels = Levels::init(set).fill();
        let mut cur = Vec::new();
        let mut next = Vec::new();

        for level in levels.iter().rev() {
            cur.clear();

            for range in Levels::child_iter_range(level) {
                let start = range.start;
                if range.is_empty() {
                    cur.push(BitVec::new());
                } else {
                    unsafe { next.get_unchecked_mut(range.clone()) }.sort_unstable();

                    // Reuse buffer. Add enclosing parentheses.
                    let fst = unsafe { next.get_unchecked_mut(start) };
                    let mut buf: BitVec<_, _> = std::mem::take(fst);

                    // Closing parenthesis.
                    buf.push(false);
                    buf.push(false);
                    buf.shift_right(1);
                    buf.set(0, true);
                    // Opening parenthesis.

                    for set in unsafe { next.get_unchecked_mut(range) }.iter().skip(1) {
                        buf.push(true);
                        buf.extend(set);
                        buf.push(false);
                    }
                    cur.push(buf);
                }
            }

            std::mem::swap(&mut cur, &mut next);
        }

        // Safety: the top level of our Levels has a root node.
        Self(unsafe { next.pop().unwrap_unchecked() })
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

// -------------------- Set algorithms -------------------- //

/// The return value for [`Levels::mod_ahu`].
#[derive(Clone, Default, Debug)]
pub struct ModAhu {
    /// Unique identifiers for the elements at the specified level.
    ///
    /// The name refers to the fact that you most often use this to work with the sets one step up,
    /// thus making this the next level of sets.
    pub next: Vec<usize>,
    /// A reusable buffer, empty when returned.
    pub buffer: Vec<usize>,
    /// The structure storing the sets and their indices.
    ///
    /// This won't be cleared by default, so you can find e.g. the largest index in `next`.
    pub sets: BTreeMap<SmallVec<usize>, usize>,
}

impl<'a> Levels<&'a Mset> {
    /// Modified [`Ahu`] algorithm. Computes a list of integers representing the distinct elements
    /// of the elements in the set at a certain level.
    ///
    /// As a small optimization, the second vector in the return value is an empty buffer that can
    /// be reused.
    #[must_use]
    pub fn mod_ahu(&self, r: usize) -> ModAhu {
        let mut cur = Vec::new();
        if self.level_len() <= r {
            return ModAhu::default();
        }
        let mut next = vec![0; self.last().len()];

        let mut sets = BTreeMap::new();
        for level in self.iter().skip(r).rev().skip(1) {
            sets.clear();
            cur.clear();

            for slice in unsafe { Levels::child_iter_mut(level, &mut next) } {
                slice.sort_unstable();
                cur.push(btree_index(
                    &mut sets,
                    slice.iter().copied().collect::<SmallVec<_>>(),
                ));
            }

            std::mem::swap(&mut cur, &mut next);
        }

        cur.clear();
        ModAhu {
            next,
            buffer: cur,
            sets,
        }
    }

    /// Returns whether `self` is a subset of `other`, meaning it contains each set at least as many
    /// times.
    ///
    /// The functions `fst_fun` and `snd_fun` update our data structure for every element found in
    /// the first and second sets respectively, and return an assigned index uniquely representing
    /// the element. Note that the second set is searched before the first. If `fst_fun` returns
    /// `None`, it means we've shown we don't have a subset.
    ///
    /// ## Precalculations
    ///
    /// It can save a lot of time to first perform basic checks as the levels are built. For
    /// instance, if some level of `self` has more elements than the corresponding level of `other`,
    /// it can't be a subset. Likewise, if all elements have the same number of elements, the subset
    /// relation actually implies equality.
    ///
    /// Calling this function implies these basic tests have already been performed. In particular,
    /// the function does not consider the case where `self` has a larger rank than `other`.
    #[must_use]
    pub(crate) fn subset_gen<
        T,
        F: FnMut(&mut BTreeMap<SmallVec<usize>, T>, SmallVec<usize>) -> Option<usize>,
        G: FnMut(&mut BTreeMap<SmallVec<usize>, T>, SmallVec<usize>) -> usize,
    >(
        &'a self,
        other: &'a Self,
        mut fst_fun: F,
        mut snd_fun: G,
    ) -> bool {
        debug_assert!(
            self.level_len() <= other.level_len(),
            "this check should have been performed beforehand"
        );

        let mut fst_cur = Vec::new();
        let mut fst_next = Vec::new();
        let mut snd_cur = Vec::new();
        let mut snd_next = Vec::new();

        let mut sets = BTreeMap::new();
        for r in (1..other.level_len()).rev() {
            sets.clear();
            fst_cur.clear();
            snd_cur.clear();

            let fst_level = self.get(r).unwrap_or_default();
            let snd_level = unsafe { other.get(r).unwrap_unchecked() };

            // Processs second set.
            for slice in Levels::child_iter(snd_level, &snd_next) {
                let mut children: SmallVec<_> = slice.iter().copied().collect();
                children.sort_unstable();
                snd_cur.push(snd_fun(&mut sets, children));
            }

            // Process first set.
            for slice in Levels::child_iter(fst_level, &fst_next) {
                let mut children: SmallVec<_> = slice.iter().copied().collect();
                children.sort_unstable();

                if let Some(idx) = fst_fun(&mut sets, children) {
                    fst_cur.push(idx);
                } else {
                    return false;
                }
            }

            std::mem::swap(&mut fst_cur, &mut fst_next);
            std::mem::swap(&mut snd_cur, &mut snd_next);
        }

        true
    }
}
