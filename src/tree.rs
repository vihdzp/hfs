use crate::prelude::*;
use std::{
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ops::Range,
};

fn vec_uninit<U>(len: usize) -> Vec<MaybeUninit<U>> {
    (0..len).map(|_| MaybeUninit::uninit()).collect()
}

/// Represents the multisets at each rank within an [`Mset`], and assigns some data to each.
///
/// To save on allocations, we use a single vector and an "indexing" vector to get subslices of it,
/// but morally, this is a `Vec<[T]>`.
///
/// ## Invariants
///
/// Every [`Levels`] must have a root level, and no level can be empty.
pub struct Levels<T> {
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
    /// In any multiset, this is guaranteed to be at least one.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// The total amount of data stored within all levels.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get the range within the slice corresponding to some level.
    pub fn get_range(&self, level: usize) -> Option<Range<usize>> {
        let start = *self.indices.get(level)?;
        let end = self.indices.get(level + 1).copied().unwrap_or(self.size());
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

    /// Returns one minus the length.
    pub fn len_pred(&self) -> usize {
        let len = self.len();
        debug_assert_ne!(len, 0);
        unsafe { len.unchecked_sub(1) }
    }

    /// Returns the last element in `indices`.
    fn last_idx(&self) -> usize {
        let last = self.indices.last();
        debug_assert!(last.is_some());
        unsafe { *last.unwrap_unchecked() }
    }

    /// Returns a reference to the last level.
    pub fn last(&self) -> &[T] {
        let idx = self.last_idx();
        &self.data[idx..]
    }

    /// Returns a mutable reference to the last level.
    pub fn last_mut(&mut self) -> &mut [T] {
        let idx = self.last_idx();
        &mut self.data[idx..]
    }

    /// Builds the next level from the previous. Returns whether the level is empty.
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
        self.indices.push(end);
        self.data.len() == end
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
    /// - `cb`: callback called on corresponding levels, makes the function return `None` if it
    ///   returns `false`
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

    /// Maps a level by a function, given the maps for the next level.
    ///
    /// - `T`: pointer type to a set-like object
    /// - `r`: current level
    /// - `cur`: current buffer to write output
    /// - `next`: the maps for the next level
    /// - `card`: a function accurately giving the number of children of `T`
    /// - `map_fn`: the mapping function
    ///
    /// ## Safety
    ///
    /// The cardinality function must be such that the sum of the cardinalities in this level equals
    /// the number of elements in the next. Otherwise, there might be uninitialized array entries,
    /// or out of bound accesses.
    ///
    /// Moreover, `r` should refer to a valid level, and `cur` should have the same size as this
    /// level.
    pub unsafe fn map_step<U, F: FnMut(&T) -> usize, G: FnMut(&[U]) -> U>(
        &self,
        r: usize,
        cur: &mut [MaybeUninit<U>],
        next: &[U],
        mut card: F,
        mut map_fn: G,
    ) {
        debug_assert!(r < self.len());
        let level = unsafe { self.get(r).unwrap_unchecked() };
        debug_assert_eq!(cur.len(), level.len());
        let mut child_start = 0;

        for (i, set) in level.iter().enumerate() {
            // Safety: we trust that the cardinality is correct.
            let child_end = child_start + card(set);
            debug_assert!(child_end <= next.len());
            let children = unsafe { next.get_unchecked(child_start..child_end) };

            let entry = unsafe { cur.get_unchecked_mut(i) };
            entry.write(map_fn(children));
            child_start = child_end;
        }
    }

    /// Maps a level by a function, given the maps for the next level.
    ///
    /// - `T`: pointer type to a set-like object
    /// - `r`: current level
    /// - `cur`: current buffer to write output
    /// - `next`: the maps for the next level
    /// - `card`: a function accurately giving the number of children of `T`
    /// - `map_fn`: the mapping function
    ///
    /// ## Safety
    ///
    /// The cardinality function must be such that the sum of the cardinalities in this level equals
    /// the number of elements in the next. Otherwise, there might be uninitialized array entries,
    /// or out of bound accesses.
    ///
    /// Moreover, `r` should refer to a valid level, and `cur` should have the same size as this
    /// level.
    pub unsafe fn map_step_mut<U, F: FnMut(&T) -> usize, G: FnMut(&mut [U]) -> U>(
        &self,
        r: usize,
        cur: &mut [MaybeUninit<U>],
        next: &mut [U],
        mut card: F,
        mut map_fn: G,
    ) {
        debug_assert!(r < self.len());
        let level = unsafe { self.get(r).unwrap_unchecked() };
        debug_assert_eq!(cur.len(), level.len());
        let mut child_start = 0;

        for (i, set) in level.iter().enumerate() {
            // Safety: we trust that the cardinality is correct.
            let child_end = child_start + card(set);
            debug_assert!(child_end <= next.len());
            let children = unsafe { next.get_unchecked_mut(child_start..child_end) };

            let entry = unsafe { cur.get_unchecked_mut(i) };
            entry.write(map_fn(children));
            child_start = child_end;
        }
    }

    /// Builds a new [`Levels`] with the same shape. Each entry is mapped to a function of its
    /// mapped children.
    ///
    /// - `T`: pointer type to a set-like object
    /// - `card`: a function accurately giving the number of children of `T`
    /// - `map_fn`: the mapping function
    ///
    /// ## Safety
    ///
    /// The cardinality function must be such that the sum of the cardinalities in one level equals
    /// the number of elements in the next. Otherwise, there might be uninitialized array entries,
    /// or out of bound accesses.
    pub unsafe fn map<U, F: FnMut(&T) -> usize, G: FnMut(&[U]) -> U>(
        &self,
        mut card: F,
        mut map_fn: G,
    ) -> Levels<U> {
        let size = self.size();
        let mut data = ManuallyDrop::new(
            (0..size)
                .map(|_| MaybeUninit::<U>::uninit())
                .collect::<Vec<_>>(),
        );

        for r in (0..self.len()).rev() {
            // Safety: We're iterating over ranks.
            let range = unsafe { self.get_range(r).unwrap_unchecked() };
            // The starting position for the children of the next node to read.
            let mut start_next = range.end;

            for i in range {
                // Safety: we trust that the cardinality is correct.
                let end_next = start_next + card(&self.data[i]);
                let children = unsafe {
                    &*(data.get_unchecked(start_next..end_next) as *const _ as *const [U])
                };

                data[i] = MaybeUninit::new(map_fn(children));
                start_next = end_next;
            }
        }

        let data = unsafe {
            Vec::from_raw_parts(data.as_mut_ptr() as *mut _, data.len(), data.capacity())
        };
        Levels {
            indices: self.indices.clone(),
            data,
        }
    }
}

impl<'a> Levels<&'a Mset> {
    /// Initializes levels for a [`Mset`].
    pub fn new(set: &'a Mset) -> Self {
        Self::new_gen(set, |buf, set| buf.extend(set))
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

macro_rules! traits {
    ($t: ty) => { impl Iterator<Item = $t> + DoubleEndedIterator + ExactSizeIterator + '_ }
}

impl<T> Levels<T> {
    pub fn iter(&self) -> traits!(&[T]) {
        (0..self.len()).map(|r| unsafe { self.get(r).unwrap_unchecked() })
    }

    pub fn iter_mut(&mut self) -> traits!(&mut [T]) {
        // Safety: these slices are all disjoint.
        let indices = &self.indices;
        let ptr = self.data.as_mut_ptr();
        (0..self.len()).map(move |r| unsafe {
            let start = indices.get_unchecked(r);
            let end = indices.get(r + 1).copied().unwrap_or(indices.len());
            std::slice::from_raw_parts_mut(ptr.add(*start), end - start)
        })
    }
}

pub fn child_iter<T, F: 'static + FnMut(&T) -> usize>(
    level: &[T],
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
            for range in child_iter(level, |s| s.card()) {
                let start = range.start;
                if !range.is_empty() {
                    next[range.clone()].sort_unstable();

                    // Reuse buffer.
                    let mut buf: BitVec<_, _> = std::mem::take(&mut next[start]);
                    buf.push(false);
                    buf.push(false);
                    buf.shift_right(1);
                    buf.set(0, true);

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
