//! Utility types and algorithms for working with [`Mset`].
//!
//! Most of the content in this file is only used internally.

use crate::prelude::*;
use std::ops::Range;

/// Represents the multisets at each rank within an [`Mset`], and assigns some data to each.
///
/// To save on allocations, we use a single vector and an "indexing" vector to get subslices of it,
/// but morally, this is a `Vec<Vec<T>>`.
///
/// ## Invariants
///
/// Every [`Levels`] must have a root level with a single node, and no level can be empty.
pub(crate) struct Levels<T> {
    /// The i-th element of the array represents the start point for the i-th level in the data
    /// array.
    indices: SmallVec<usize>,

    /// Stores all the data for all levels.
    data: Vec<T>,
}

impl<T> Levels<T> {
    /// Initializes the first level from a set.
    pub fn init(set: T) -> Self {
        Self {
            indices: smallvec::smallvec![0],
            data: vec![set],
        }
    }

    /// The number of levels stored.
    ///
    /// This is actually the rank of the multiset plus one.
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// The total amount of data stored within all levels.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get the range within the slice corresponding to some level.
    pub fn get_range(&self, level: usize) -> Option<Range<usize>> {
        let start = *self.indices.get(level)?;
        let end = self.indices.get(level + 1).copied().unwrap_or(self.len());
        Some(start..end)
    }

    /// Gets the slice corresponding to a given level.
    pub fn get(&self, level: usize) -> Option<&[T]> {
        self.get_range(level).map(|range| &self.data[range])
    }

    /// Gets the mutable slice corresponding to a given level.
    pub fn get_mut<'a>(&'a mut self, level: usize) -> Option<&'a mut [T]> {
        self.get_range(level).map(|range| &mut self.data[range])
    }

    /// Returns the last element in `indices`.
    pub fn last_idx(&self) -> usize {
        unsafe { *self.indices.last().unwrap_unchecked() }
    }

    /// Returns the last level.
    pub fn last(&self) -> &[T] {
        unsafe { self.data.get_unchecked(self.last_idx()..) }
    }

    /// Builds the next level from the last. Returns whether this level is empty.
    ///
    /// - `T`: pointer type to a set-like object
    /// - `extend`: a function extending an array with the children of a set `T`
    /// - `buf`: a buffer for calculations.
    pub fn step<F: FnMut(&mut Vec<T>, T)>(&mut self, mut extend: F, buf: &mut Vec<T>) -> bool
    where
        T: Copy,
    {
        // Gets the last level.
        let start = self.last_idx();
        let end = self.data.len();

        // Adds elements of each set in the last level.
        for i in start..end {
            let set = self.data[i];
            extend(buf, set);
            self.data.extend(&*buf);
            buf.clear();
        }

        // Return whether the level is empty.
        let finish = self.data.len() == end;
        if !finish {
            self.indices.push(end);
        }
        finish
    }

    /// A generic procedure to build [`Levels`].
    ///
    /// See [`Self::new`] and [`Self::new_mut`].
    ///
    /// - `T`: pointer type to a set-like object
    /// - `extend`: a function extending an array with the children of a set `T`
    pub fn new_gen<F: FnMut(&mut Vec<T>, T)>(set: T, mut extend: F) -> Self
    where
        T: Copy,
    {
        let mut levels = Self::init(set);
        let mut buf = Vec::new();
        while !levels.step(&mut extend, &mut buf) {}
        levels
    }

    /// Initializes two [`Levels`] simultaneously. Calls a function on every pair of levels built,
    /// which determines whether execution is halted early.
    ///
    /// - `T`: pointer type to a set-like object
    /// - `extend`: a function extending an array with the children of a set `T`
    /// - `cb`: callback called on corresponding levels, as they're built, makes the function return
    ///   `None` if it returns `false`
    pub fn both<F: FnMut(&mut Vec<T>, T), G: FnMut(&[T], &[T]) -> bool>(
        set: T,
        other: T,
        mut extend: F,
        mut cb: G,
    ) -> Option<(Self, Self)>
    where
        T: Copy,
    {
        let mut fst = Self::init(set);
        let mut snd = Self::init(other);
        let mut finish_fst = false;
        let mut finish_snd = false;
        let mut level = 1;
        let mut buf = Vec::new();

        loop {
            // Step execution.
            if !finish_fst {
                finish_fst = fst.step(&mut extend, &mut buf);
            }
            if !finish_snd {
                finish_snd = snd.step(&mut extend, &mut buf);
            }

            // Check if finished.
            if !cb(fst.get(level).unwrap_or(&[]), snd.get(level).unwrap_or(&[])) {
                return None;
            }
            if finish_fst && finish_snd {
                return Some((fst, snd));
            }
            level += 1;
        }
    }
}

macro_rules! traits {
    ($t: ty) => { impl Iterator<Item = $t> + DoubleEndedIterator + ExactSizeIterator + '_ }
}

impl<T> Levels<T> {
    pub fn iter(&self) -> traits!(&[T]) {
        (0..self.rank()).map(|r| unsafe { self.get(r).unwrap_unchecked() })
    }

    pub fn iter_mut(&mut self) -> traits!(&mut [T]) {
        // Safety: these slices are all disjoint.
        let indices = &self.indices;
        let ptr = self.data.as_mut_ptr();
        (0..self.rank()).map(move |r| unsafe {
            let start = indices.get_unchecked(r);
            let end = indices.get(r + 1).copied().unwrap_or(indices.len());
            std::slice::from_raw_parts_mut(ptr.add(*start), end - start)
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

            while let Some(next) = iter.next() {
                write!(f, " | {next}")?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

impl<'a> Levels<&'a Mset> {
    /// Initializes levels for a [`Mset`].
    pub fn new(set: &'a Mset) -> Self {
        Self::new_gen(set, Vec::extend)
    }

    /// For each set in a level within [`Levels`], finds the range for its children in the next
    /// level.
    pub fn child_iter(level: &'a [&'a Mset]) -> traits!(Range<usize>) {
        Self::child_iter_gen(level, |s| s.card())
    }

    /// Returns whether `self` is a subset of `other`, meaning it contains each set at least as many
    /// times. We check this through a modified [`Ahu`] algorithm.
    pub fn subset(&'a self, other: &'a Self) -> bool {
        let mut fst_cur = Vec::new();
        let mut fst_next = Vec::new();
        let mut snd_cur = Vec::new();
        let mut snd_next = Vec::new();

        // Each set gets assigned a unique integer, and a "weighted count" of times found in
        // `other` minus times found in `self`.
        //
        // If this weighted count goes into the negatives, return false.
        let mut sets = BTreeMap::new();

        for r in (1..other.rank()).rev() {
            fst_cur.clear();
            snd_cur.clear();

            let fst_level = self.get(r).unwrap_or_default();
            let snd_level = unsafe { other.get(r).unwrap_unchecked() };

            for snd_range in Self::child_iter(snd_level) {
                let range = unsafe { snd_next.get_unchecked(snd_range).iter().copied() };
                let mut children = SmallVec::from_iter(range);
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
                let range = unsafe { fst_next.get_unchecked(fst_range).iter().copied() };
                let mut children = SmallVec::from_iter(range);
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

            sets.clear();
            std::mem::swap(&mut fst_cur, &mut fst_next);
            std::mem::swap(&mut snd_cur, &mut snd_next);
        }

        true
    }
}

impl<'a> Levels<*mut Mset> {
    /// Initializes mutable levels for a [`Mset`]. Pointers are reqiured as each level mutably
    /// aliases the next.
    ///
    /// ## Safety
    ///
    /// This method is completely safe, but you must be careful dereferencing pointers. Modifying a
    /// set and trying to access its children will often result in an invalid dereference.
    pub fn new_mut(set: &'a mut Mset) -> Self {
        // The set is not mutated, so the pointers remain valid to dereference.
        Self::new_gen(set, |buf, set| {
            buf.extend(unsafe { &mut *set }.iter_mut().map(|x| x as *mut _))
        })
    }
}

/// The [Aho–Hopcroft–Ullman](https://www.baeldung.com/cs/isomorphic-trees) (AHU) encoding for an
/// [`Mset`]. It is unique up to multiset equality.
///
/// Conceptually, this amounts to hereditarily lexicographically ordered set-builder notation. In
/// fact, the [`Display`] implementation for [`Mset`] constructs an [`Ahu`] first.
///
/// The issue with this encoding is that after the first few levels, it becomes expensive to store
/// and compare all of the partial encodings. As such, instead of computing the full AHU encoding,
/// we often opt for a modified encoding, where at each step, each unique multiset is assigned a
/// single integer instead of the full string. This "modified" AHU encoding does not determine
/// multisets uniquely, but it can uniquely determine multisets within a single multiset.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, IntoIterator)]
pub struct Ahu(#[into_iterator(owned, ref)] BitVec);

impl Ahu {
    /// The empty encoding.
    pub const fn empty() -> Self {
        Self(BitVec::EMPTY)
    }

    /// Finds the AHU encoding for a multiset.
    pub fn new(set: &Mset) -> Self {
        let levels = Levels::new(set);
        let mut cur = Vec::new();
        let mut next = Vec::new();

        for level in levels.iter().rev() {
            cur.clear();
            for range in Levels::child_iter(level) {
                let start = range.start;
                if !range.is_empty() {
                    next[range.clone()].sort_unstable();

                    // Reuse buffer.
                    // Add enclosing parentheses.
                    let mut buf: BitVec<_, _> = std::mem::take(&mut next[start]);
                    buf.push(false); // )
                    buf.push(false);
                    buf.shift_right(1);
                    buf.set(0, true); // (

                    for set in next[range].iter().skip(1) {
                        buf.push(true);
                        buf.extend(set);
                        buf.push(false);
                    }
                    cur.push(buf)
                } else {
                    cur.push(BitVec::new());
                }
            }

            std::mem::swap(&mut cur, &mut next);
        }

        Self(next.pop().unwrap())
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
