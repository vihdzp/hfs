//! Hereditarily finite multisets [`Mset`].

use crate::prelude::*;

/// A [hereditarily finite](https://en.wikipedia.org/wiki/Hereditarily_finite_set)
/// [multiset](https://en.wikipedia.org/wiki/Multiset).
///
/// Unlike [sets](Set), multisets can contain an element multiple times. The number of times they do
/// is referred to as their multiplicity or [count](Mset::count). Familiar operations on sets, like
/// unions and intersections, are reinterpreted in terms of these multiplicities.
///
/// ## Internal representation
///
/// Each [`Mset`] contains only a `Vec` of [`Mset`]. Rust's ownership system guarantees that [quine
/// atoms](https://en.wikipedia.org/wiki/Urelement#Quine_atoms), or any other non-regular multisets
/// cannot exist. In other words, a multiset can't contain itself.
#[derive(Clone, Default, Eq, IntoIterator)]
pub struct Mset(#[into_iterator(owned, ref, ref_mut)] pub Vec<Mset>);

// -------------------- Basic traits -------------------- //

impl AsRef<Mset> for Mset {
    fn as_ref(&self) -> &Mset {
        self
    }
}

impl From<Mset> for Vec<Mset> {
    fn from(set: Mset) -> Self {
        set.0
    }
}

impl From<Vec<Mset>> for Mset {
    fn from(vec: Vec<Mset>) -> Self {
        Self(vec)
    }
}

impl FromIterator<Mset> for Mset {
    fn from_iter<T: IntoIterator<Item = Mset>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// Succintly writes a multiset as is stored in memory.
impl Debug for Mset {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.write_char('(')?;
        for el in self {
            write!(f, "{el:?}")?;
        }
        f.write_char(')')
    }
}

/// Displays a multiset in canonical roster notation.
impl Display for Mset {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self.ahu())
    }
}

impl PartialEq for Mset {
    fn eq(&self, other: &Self) -> bool {
        // Don't even allocate if not needed.
        if self.card() != other.card() {
            return false;
        }

        if let Some((fst, snd)) = self.eq_levels(other) {
            fst.subset(&snd)
        } else {
            false
        }
    }
}

impl PartialOrd for Mset {
    fn le(&self, other: &Self) -> bool {
        // Don't even allocate if not needed.
        if self.card() > other.card() {
            return false;
        }

        if let Some((fst, snd)) = self.le_levels(other) {
            fst.subset(&snd)
        } else {
            false
        }
    }

    fn ge(&self, other: &Self) -> bool {
        other.le(self)
    }

    fn lt(&self, other: &Self) -> bool {
        // Don't even allocate if not needed.
        if self.card() >= other.card() {
            return false;
        }

        if let Some((fst, snd)) = self.le_levels(other) {
            fst.subset(&snd)
        } else {
            false
        }
    }

    fn gt(&self, other: &Self) -> bool {
        other.lt(self)
    }

    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
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

// -------------------- String parsing -------------------- //

/// Error in parsing a set. This can only happen due to mismatched brackets.
#[derive(Clone, Copy, Debug)]
pub struct SetError;

impl Display for SetError {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.write_str("mismatched brackets")
    }
}

impl std::error::Error for SetError {}

/// Multisets are parsed from their roster notation. Any symbol other than `{` and `}` is ignored,
/// including commas.
impl FromStr for Mset {
    type Err = SetError;

    fn from_str(s: &str) -> Result<Self, SetError> {
        let mut stack = Vec::new();
        let mut iter = s.chars();

        loop {
            let c = iter.next().ok_or(SetError)?;
            match c {
                // New multiset.
                '{' => stack.push(Self::empty()),

                // Close last multiset.
                '}' => {
                    let last = stack.pop().ok_or(SetError)?;
                    if let Some(prev) = stack.last_mut() {
                        prev.insert_mut(last);
                    } else {
                        // Multiset has been built.
                        for c in iter {
                            if ['{', '}'].contains(&c) {
                                return Err(SetError);
                            }
                        }

                        return Ok(last);
                    }
                }
                _ => {}
            }
        }
    }
}

// -------------------- SetTrait -------------------- //

impl crate::Seal for Mset {}

impl SetTrait for Mset {
    // -------------------- Basic methods -------------------- //

    fn as_slice(&self) -> &[Self] {
        &self.0
    }

    unsafe fn _as_mut_slice(&mut self) -> &mut [Self] {
        &mut self.0
    }

    fn as_vec(&self) -> &Vec<Mset> {
        &self.0
    }

    unsafe fn _as_mut_vec(&mut self) -> &mut Vec<Mset> {
        &mut self.0
    }

    fn _flatten_vec(vec: Vec<Self>) -> Self {
        vec.into()
    }

    // -------------------- Constructions -------------------- //

    fn empty() -> Self {
        Self(Vec::new())
    }

    fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    fn singleton(self) -> Self {
        Self(vec![self])
    }

    fn into_singleton(mut self) -> Option<Self> {
        if self.card() != 1 {
            return None;
        }

        self.0.pop()
    }

    fn insert_mut(&mut self, other: Self) {
        self.0.push(other);
    }

    fn select_mut<P: FnMut(&Mset) -> bool>(&mut self, mut pred: P) {
        let mut i = 0;
        while i < self.card() {
            if pred(&self.0[i]) {
                i += 1;
            } else {
                self.0.swap_remove(i);
            }
        }
    }

    fn count(&self, set: &Self) -> usize {
        let mut cmp = Compare::new(set);
        self.iter().filter(|el| cmp.eq(el)).count()
    }

    fn sum(mut self, mut other: Self) -> Self {
        self.0.append(&mut other.0);
        self
    }

    fn sum_vec(vec: Vec<Self>) -> Self {
        Self::sum_iter(vec)
    }

    fn union_vec(mut vec: Vec<Self>) -> Self {
        // Check for trivial cases.
        match vec.len() {
            0 => return Self::empty(),
            1 => return vec.pop().unwrap(),
            _ => {}
        }

        let levels = Levels::new_iter(&vec);
        let next = levels.ahu(1);
        // Safety: the length of `next` is exactly the sum of cardinalities in the first level.
        let mut iter = unsafe { levels.children_slice(0, &next) }.enumerate();

        // Each entry stores indices to where it's found in `vec`, and a counter for how many times
        // it's been seen in every other set.
        //
        // It should be possible to use a single pointer instead of two indices. But I've got no
        // idea how to make this work safely.
        //
        // Safety: we already know there's at least 2 sets.
        let fst = unsafe { iter.next().unwrap_unchecked().1 };
        let mut sets = BTreeMap::new();
        for (i, &set) in fst.iter().enumerate() {
            let el_idx = (0, i);
            match sets.entry(set) {
                Entry::Vacant(entry) => {
                    entry.insert((smallvec![el_idx], 0));
                }
                Entry::Occupied(mut entry) => {
                    entry.get_mut().0.push(el_idx);
                }
            }
        }

        // Count number of appearances in other sets.
        for (n, slice) in iter {
            for (i, &set) in slice.iter().enumerate() {
                let el_idx = (n, i);
                match sets.entry(set) {
                    Entry::Vacant(entry) => {
                        entry.insert((smallvec![el_idx], 1));
                    }
                    Entry::Occupied(mut entry) => {
                        let (indices, count) = entry.get_mut();
                        if indices.len() == *count {
                            indices.push(el_idx);
                        }
                        *count += 1;
                    }
                }
            }

            // Reset counts.
            for (_, count) in sets.values_mut() {
                *count = 0;
            }
        }

        // Get all the sets we need.
        let mut union = Self::empty();
        for (indices, _) in sets.into_values() {
            for (n, i) in indices {
                // Safety: all the indices we built are valid for this sort of indexing.
                let set = mem::take(unsafe { vec.get_unchecked_mut(n).0.get_unchecked_mut(i) });
                union.insert_mut(set);
            }
        }

        union
    }

    fn inter_vec(mut vec: Vec<Self>) -> Option<Self> {
        // Check for trivial cases.
        match vec.len() {
            0 => return None,
            1 => return Some(vec.pop().unwrap()),
            _ => {}
        }
        let levels = Levels::new_iter(&vec);

        let next = levels.ahu(1);
        // Safety: the length of `next` is exactly the sum of cardinalities in the first level.
        let mut iter = unsafe { levels.children_slice(0, &next) };

        // Each entry stores the indices where it's found within the first set, and a counter for
        // how many times it's been seen in every other set.
        //
        // Safety: we already know there's at least 2 sets.
        let fst = unsafe { iter.next().unwrap_unchecked() };
        let mut sets = BTreeMap::new();
        for (i, &set) in fst.iter().enumerate() {
            match sets.entry(set) {
                Entry::Vacant(entry) => {
                    entry.insert((smallvec![i], 0));
                }
                Entry::Occupied(mut entry) => {
                    entry.get_mut().0.push(i);
                }
            }
        }

        // Count number of appearances in other sets.
        for slice in iter {
            for &set in slice {
                match sets.entry(set) {
                    Entry::Vacant(_) => {}
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().1 += 1;
                    }
                }
            }

            // Update counts.
            sets.retain(|_, (indices, count)| {
                indices.truncate(*count);
                let retain = *count != 0;
                *count = 0;
                retain
            });
        }

        // Take elements from the first set, reuse some other set as a buffer.
        let mut fst = vec.swap_remove(0);
        let mut snd = vec.swap_remove(0);
        snd.clear();

        for (indices, _) in sets.into_values() {
            for i in indices {
                // Safety: all the indices we built are valid for the first set.
                let set = mem::take(unsafe { fst.0.get_unchecked_mut(i) });
                snd.insert_mut(set);
            }
        }

        Some(snd)
    }

    fn powerset(self) -> Self {
        // P(Ø) = {Ø}.
        let n = self.card();
        let mut powerset = Self::empty().singleton();
        if n == 0 {
            return powerset;
        }

        // Subsets are in correspondence to bitmasks.
        for mut i in 1..((1 << n) - 1) {
            let mut subset = Self::empty();
            for j in 0..n {
                if i % 2 == 1 {
                    subset.insert_mut(self.0[j].clone());
                }
                i /= 2;
            }

            powerset.insert_mut(subset);
        }

        powerset.insert(self)
    }

    fn nat(n: usize) -> Self {
        let mut res = Mset::empty();
        for _ in 0..n {
            res.insert_mut(res.clone());
        }
        res
    }

    fn zermelo(n: usize) -> Self {
        let mut res = Mset::empty();
        for _ in 0..n {
            res = res.singleton();
        }
        res
    }

    fn neumann(n: usize) -> Self {
        debug_assert!(
            n <= 5,
            "the sixth set in the von Neumann hierarchy has 2^65536 elements"
        );

        let mut set = Self::empty();
        for _ in 0..n {
            set = set.powerset();
        }
        set
    }

    // -------------------- Relations -------------------- //

    // See [`Set::inter_vec`].
    fn disjoint_iter<'a, I: IntoIterator<Item = &'a Self>>(iter: I) -> bool {
        let levels = Levels::new_iter(iter);

        // Check for trivial cases.
        let slice = levels.nest_vec().get(0);
        match slice.len() {
            // The empty family is not disjoint.
            0 => return false,
            // A singleton {x} is only disjoint when x = Ø.
            1 => return levels.nest_vec().len() == 1,
            _ => {}
        }

        let next = levels.ahu(1);
        // Safety: the length of `next` is exactly the sum of cardinalities in the first level.
        let mut iter = unsafe { levels.children_slice(0, &next) };

        // Each entry stores a boolean for whether it's been seen in every other set.
        //
        // Safety: we already know there's at least 2 sets.
        let fst = unsafe { iter.next().unwrap_unchecked() };
        let mut sets = BTreeMap::new();
        for &set in fst {
            sets.insert(set, false);
        }

        // Look for appearances in other sets.
        for slice in iter {
            for &set in slice {
                if let Entry::Occupied(mut entry) = sets.entry(set) {
                    *entry.get_mut() = true;
                }
            }

            // Update counts.
            sets.retain(|_, count| {
                let retain = *count;
                *count = false;
                retain
            });

            if sets.is_empty() {
                return true;
            }
        }

        false
    }

    // See [`Mset::union_vec`].
    fn disjoint_pairwise<'a, I: IntoIterator<Item = &'a Self>>(iter: I) -> bool {
        let levels = Levels::new_iter(iter);
        // Check for trivial cases.
        if levels.nest_vec().get(0).len() <= 1 {
            return true;
        }

        let next = levels.ahu(1);
        // Safety: the length of `next` is exactly the sum of cardinalities in the first level.
        let mut iter = unsafe { levels.children_slice(0, &next).enumerate() };

        // Each entry stores the index of the set in which it's found.
        //
        // Safety: we already know there's at least 2 sets.
        let (_, fst) = unsafe { iter.next().unwrap_unchecked() };
        let mut sets = BTreeMap::new();
        for &set in fst {
            sets.insert(set, 0);
        }

        // Look for appearances in other sets.
        for (i, slice) in iter {
            for &set in slice {
                match sets.entry(set) {
                    Entry::Vacant(entry) => {
                        entry.insert(i);
                    }

                    Entry::Occupied(entry) => {
                        if *entry.get() != i {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    // -------------------- Axioms -------------------- //

    fn into_choose(mut self) -> Option<Self> {
        if self.is_empty() {
            None
        } else {
            Some(self.0.swap_remove(0))
        }
    }

    fn choose_uniq(&self) -> Option<&Self> {
        self.choose_uniq_idx().map(
            // Safety: our index is valid.
            |idx| unsafe { self.0.get_unchecked(idx) },
        )
    }

    fn into_choose_uniq(mut self) -> Option<Self> {
        self.choose_uniq_idx()
            // Safety: our index is valid.
            .map(|idx| unsafe { mem::take(self.0.get_unchecked_mut(idx)) })
    }
}

// -------------------- Mset specific -------------------- //

impl Mset {
    /// The set as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [Self] {
        &mut self.0
    }

    /// A mutable reference to the inner vector.
    pub fn as_mut_vec(&mut self) -> &mut Vec<Self> {
        &mut self.0
    }

    /// Mutably iterate over the elements of the set.
    pub fn iter_mut(&mut self) -> slice::IterMut<Self> {
        self.0.iter_mut()
    }

    /// Whether the multiset contains duplicate elements.
    #[must_use]
    pub fn duplicate(&self) -> bool {
        Levels::new(self).duplicate(1)
    }

    /// Sum over an iterator.
    ///
    /// This is not implemented for [`Set`], as the method would require directly collecting the
    /// iterator into a vector anyways.
    pub fn sum_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter().flatten().collect()
    }

    /// Chooses some arbitrary index containing an element. Equal multisets should get assigned
    /// indices that correspond to equal multisets.
    fn choose_uniq_idx(&self) -> Option<usize> {
        // Select the element with the lowest AHU encoding.
        Ahu::new_iter(self)
            .iter()
            .enumerate()
            .min_by_key(|s| s.1)
            .map(|(idx, _)| idx)
    }
}
