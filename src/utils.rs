//! Utility types and algorithms for working with sets.

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

/// Represents the multisets at each rank within an [`Mset`], and assigns some data to each.
///
/// To save on allocations, we use a single vector and an "indexing" vector to get subslices of it,
/// but morally, this is a `Vec<Vec<T>>`. As a further optimization, many functions take in an
/// initial [`Levels`] to use as a buffer, which can be built from [`Self::init`] or reused with
/// [`Self::init_with`].
///
/// ## Invariants
///
/// - Every [`Levels`] must have a root level with a single node, and no level can be empty.
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

impl<T> Levels<T> {
    /// Initializes an empty [`Levels`]. **This breaks type invariants**.
    ///
    /// ## Safety
    ///
    /// The only valid use for this is as a placeholder or as an initial buffer to be immediately
    /// filled.
    pub unsafe fn empty() -> Self {
        Self {
            indices: SmallVec::new(),
            data: Vec::new(),
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

    /// Initializes the first level from a set.
    pub fn init(set: T) -> Self {
        Self {
            indices: smallvec![0],
            data: vec![set],
        }
    }

    /// The number of levels stored.
    #[must_use]
    pub fn level_len(&self) -> usize {
        self.indices.len()
    }

    /// The rank of the set.
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

    /// Builds the next level from the last. Returns whether this level was nonempty.
    ///
    /// - `T`: pointer type to a set-like object
    /// - `extend`: a function extending an array with the children of a set `T`
    /// - `buf`: a buffer for calculations.
    pub fn step_gen<F: FnMut(&mut Vec<T>, T)>(&mut self, mut extend: F, buf: &mut Vec<T>) -> bool
    where
        T: Copy,
    {
        // Gets the last level.
        let start = self.last_idx();
        let end = self.data.len();

        // Adds elements of each set in the last level.
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

    /// A generic procedure to build [`Levels`].
    ///
    /// See [`Self::new`] and [`Self::new_mut`] for specific instantiations.
    ///
    /// - `T`: pointer type to a set-like object
    /// - `extend`: a function extending an array with the children of a set `T`
    #[must_use]
    pub fn new_gen<F: FnMut(&mut Vec<T>, T)>(mut self, mut extend: F) -> Self
    where
        T: Copy,
    {
        let mut buf = Vec::new();
        while self.step_gen(&mut extend, &mut buf) {}
        self
    }

    /// Initializes two [`Levels`] simultaneously. Calls a function on every pair of levels built,
    /// which determines whether execution is halted early.
    ///
    /// - `T`: pointer type to a set-like object
    /// - `extend`: a function extending an array with the children of a set `T`
    /// - `cb`: callback called on corresponding levels, as they're built, makes the function return
    ///   `None` if it returns `false`
    pub fn both_gen<F: FnMut(&mut Vec<T>, T), G: FnMut(&[T], &[T]) -> bool>(
        mut self,
        mut other: Self,
        mut extend: F,
        mut cb: G,
    ) -> Option<(Self, Self)>
    where
        T: Copy,
    {
        let mut cont_fst = false;
        let mut cont_snd = false;
        let mut level = 1;
        let mut buf = Vec::new();

        loop {
            // Step execution.
            if cont_fst {
                cont_fst = self.step_gen(&mut extend, &mut buf);
            }
            if cont_snd {
                cont_snd = other.step_gen(&mut extend, &mut buf);
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
}

/// Shorthand for the traits our iterators implement.
macro_rules! traits {
    ($t: ty) => { impl DoubleEndedIterator<Item = $t> + ExactSizeIterator + '_ }
}

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
    ) -> traits!(Range<usize>) {
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
        for (i, level) in self.iter().enumerate() {
            let mut iter = level.iter();
            write!(f, "Level {i:02}: {}", unsafe {
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
    /// Builds the next level from the last. Returns whether this level was nonempty.
    ///
    /// See [`Self::step_gen`].
    pub fn step(&mut self, buf: &mut Vec<&'a Mset>) -> bool {
        self.step_gen(Vec::extend, buf)
    }

    /// Initializes levels for a [`Mset`].
    #[must_use]
    pub fn new(self) -> Self {
        self.new_gen(Vec::extend)
    }

    /// Initializes two [`Levels`] simultaneously. Calls a function on every pair of levels built,
    /// which determines whether execution is halted early.
    ///
    /// See [`Self::both_gen`].
    pub fn both<F: FnMut(&[&'a Mset], &[&'a Mset]) -> bool>(
        self,
        other: Self,
        cb: F,
    ) -> Option<(Self, Self)> {
        self.both_gen(other, Vec::extend, cb)
    }

    /// For each set in a level within [`Levels`], finds the range for its children in the next
    /// level.
    #[must_use]
    pub fn child_iter(level: &'a [&'a Mset]) -> traits!(Range<usize>) {
        Self::child_iter_gen(level, |s| s.card())
    }

    /// Computes a list of integers representing the distinct elements of the elements in the set at
    /// a certain level.
    ///
    /// As a small optimization, the second vector in the return value is an empty buffer that can
    /// be reused.
    #[must_use]
    pub fn mod_ahu(&self, r: usize) -> (Vec<usize>, Vec<usize>) {
        let mut cur = Vec::new();
        if self.level_len() <= r {
            return (Vec::new(), cur);
        }
        let mut next = vec![0; self.last().len()];

        let mut sets = BTreeMap::new();
        for level in self.iter().skip(r).rev().skip(1) {
            sets.clear();
            cur.clear();

            for range in Levels::child_iter(level) {
                let slice = unsafe {
                    let slice = next.get_unchecked_mut(range);
                    slice.sort_unstable();
                    slice as &[_]
                };

                cur.push(btree_index(
                    &mut sets,
                    slice.iter().copied().collect::<SmallVec<_>>(),
                ));
            }

            std::mem::swap(&mut cur, &mut next);
        }

        cur.clear();
        (next, cur)
    }

    /// Returns whether `self` is a subset of `other`, meaning it contains each set at least as many
    /// times.
    ///
    /// It can save a lot of time to first perform basic checks as the levels are built. For
    /// instance, if some level of `self` has more elements than the corresponding level of `other`,
    /// it can't be a subset. Likewise, if all elements have the same number of elements, the subset
    /// relation actually implies equality.
    #[must_use]
    pub fn subset(&'a self, other: &'a Self) -> bool {
        let mut fst_cur = Vec::new();
        let mut fst_next = Vec::new();
        let mut snd_cur = Vec::new();
        let mut snd_next = Vec::new();

        // Each set gets assigned a unique integer, and a "weighted count" of times found in `other`
        // minus times found in `self`.
        //
        // If this weighted count goes into the negatives, return false.
        let mut sets = BTreeMap::new();

        for r in (1..other.level_len()).rev() {
            sets.clear();
            fst_cur.clear();
            snd_cur.clear();

            let fst_level = self.get(r).unwrap_or_default();
            let snd_level = unsafe { other.get(r).unwrap_unchecked() };

            for snd_range in Self::child_iter(snd_level) {
                let mut children: SmallVec<_> =
                    unsafe { snd_next.get_unchecked(snd_range).iter().copied() }.collect();
                children.sort_unstable();

                // Number of times found, minus one.
                let len = sets.len();
                match sets.entry(children) {
                    Entry::Vacant(entry) => {
                        entry.insert((len, 0));
                        snd_cur.push(len);
                    }
                    Entry::Occupied(mut entry) => {
                        let (idx, num) = entry.get_mut();
                        snd_cur.push(*idx);
                        *num += 1;
                    }
                }
            }

            for fst_range in Self::child_iter(fst_level) {
                let mut children: SmallVec<_> =
                    unsafe { fst_next.get_unchecked(fst_range).iter().copied() }.collect();
                children.sort_unstable();

                // Remove one to the times found.
                match sets.entry(children) {
                    Entry::Vacant(_) => return false,
                    Entry::Occupied(mut entry) => {
                        let (idx, num) = entry.get_mut();
                        fst_cur.push(*idx);
                        if *num == 0 {
                            entry.remove_entry();
                        } else {
                            *num -= 1;
                        }
                    }
                }
            }

            std::mem::swap(&mut fst_cur, &mut fst_next);
            std::mem::swap(&mut snd_cur, &mut snd_next);
        }

        true
    }
}

impl Levels<*mut Mset> {
    /// Initializes mutable levels for a [`Mset`]. Pointers are reqiured as each level mutably
    /// aliases the next.
    ///
    /// ## Safety
    ///
    /// This method is completely safe, but you must be careful dereferencing pointers. Modifying a
    /// set and trying to access its children will often result in an invalid dereference.
    #[must_use]
    pub fn new_mut(self) -> Self {
        // The set is not mutated, so the pointers remain valid to dereference.
        self.new_gen(|buf, set| {
            buf.extend(unsafe { &mut *set }.iter_mut().map(std::ptr::from_mut));
        })
    }
}

/// The [Aho–Hopcroft–Ullman](https://www.baeldung.com/cs/isomorphic-trees) (AHU) encoding for an
/// [`Mset`]. It is unique up to multiset equality.
///
/// Conceptually, this amounts to hereditarily lexicographically ordered set-builder notation. In
/// fact, the [`Display`] implementation for [`Mset`] constructs an [`Ahu`] first.
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

    /// Finds the [`Ahu`] encoding for a multiset.
    #[must_use]
    pub fn new(set: &Mset) -> Self {
        let levels = Levels::init(set).new();
        let mut cur = Vec::new();
        let mut next = Vec::new();

        for level in levels.iter().rev() {
            cur.clear();

            for range in Levels::child_iter(level) {
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

        // Safety: the root note in `Levels` always exists.
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
