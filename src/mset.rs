//! Hereditarily finite multisets [`Mset`].

use crate::prelude::*;

/// A hereditarily finite multiset.
#[derive(Clone, Default, Eq, IntoIterator)]
pub struct Mset(#[into_iterator(owned, ref, ref_mut)] pub Vec<Mset>);

impl FromIterator<Mset> for Mset {
    fn from_iter<T: IntoIterator<Item = Mset>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl Debug for Mset {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_char('(')?;
        for el in self {
            write!(f, "{el:?}")?;
        }
        f.write_char(')')
    }
}

impl Display for Mset {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.ahu())
    }
}

/// Error in parsing a multiset. This can only happen due to mismatched brackets.
#[derive(Clone, Copy, Debug)]
pub struct MsetError;

impl Display for MsetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("mismatched brackets")
    }
}

impl Error for MsetError {}

/// Sets are parsed from their set-builder notation. Any symbol other than `{` and `}` is ignored.
impl FromStr for Mset {
    type Err = MsetError;

    fn from_str(s: &str) -> Result<Self, MsetError> {
        let mut stack = Vec::new();
        let mut iter = s.chars();
        loop {
            let c = iter.next().ok_or(MsetError)?;
            match c {
                '{' => stack.push(Self::empty()),
                '}' => {
                    let last = stack.pop().ok_or(MsetError)?;
                    if let Some(prev) = stack.last_mut() {
                        prev.insert_mut(last);
                    } else {
                        // Set has been built.
                        while let Some(c) = iter.next() {
                            if ['{', '}'].contains(&c) {
                                return Err(MsetError);
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

impl Mset {
    /// A common method for determining equality / subsets.
    fn eq_aux(&self, other: &Self, eq: bool) -> bool {
        // Subdivide the nodes of each set into levels.
        // Ignore the root node.
        let mut fst = Vec::new();
        let mut snd = Vec::new();
        let mut fst_last = vec![self];
        let mut snd_last = vec![other];
        let mut top = true;
        while !snd_last.is_empty() {
            let mut fst_cur = Vec::new();
            let mut snd_cur = Vec::new();

            for &set in &fst_last {
                for el in set {
                    fst_cur.push(el);
                }
            }
            for &set in &snd_last {
                for el in set {
                    snd_cur.push(el);
                }
            }

            // Return false if two levels don't have matching numbers of elements.
            let cmp = fst_last.len().cmp(&snd_last.len());
            if (eq && cmp.is_ne()) || (!eq && cmp.is_gt()) {
                return false;
            }

            if !top {
                fst.push(fst_last);
                snd.push(snd_last);
            }
            top = false;
            fst_last = fst_cur;
            snd_last = snd_cur;
        }

        // The first set can't have a rank larger than the second.
        if !fst_last.is_empty() {
            return false;
        }
        // If the rank is at most 1, these checks are enough.
        let rank = fst_last.len();
        if rank <= 1 {
            return true;
        }

        // Given the sets from the next level (encoded as integers), finds encodings for the sets
        // in this level.
        let mut fst_next = vec![0; fst[rank - 1].len()];
        let mut snd_next = vec![0; snd[rank - 1].len()];

        // Sets found on each level.
        // Each set gets assigned a unique integer, and a "weighted count".
        let mut sets = BTreeMap::new();
        for r in (0..(rank - 1)).rev() {
            sets.clear();

            // Collect sets from snd.
            let snd_level = &snd[r];
            let size = snd_level.len();
            let mut snd_cur = Vec::with_capacity(size);

            let mut child = 0;
            for set in snd_level {
                let mut el = SmallVec::new();
                for _ in 0..set.card() {
                    el.push(snd_next[child]);
                    child += 1;
                }

                el.sort_unstable();
                let len = sets.len();
                // Increase the count for each set.
                match sets.entry(el) {
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

            // Collect sets from fst.
            let fst_level = &fst[r];
            let mut fst_cur = Vec::with_capacity(size);

            child = 0;
            for set in fst_level {
                let mut el = SmallVec::new();
                for _ in 0..set.card() {
                    el.push(fst_next[child]);
                    child += 1;
                }

                el.sort_unstable();
                // Decrease the count for each set, return false if it reaches a negative.
                match sets.entry(el) {
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

            // sets must be empty at this point.
            if eq {
                debug_assert!(sets.is_empty());
            }

            fst_next = fst_cur;
            snd_next = snd_cur;
        }

        true
    }
}

/// Two multisets are equal when they have the same elements the same amount of times.
///
/// We check this through a modified [`Ahu`] algorithm. We first grade the hereditary elements of
/// both sets. We then compute AHU encodings from the bottom up. If at any level we find a mismatch,
/// we return `false`. Otherwise, we assign integers to each of our encodings and continue the
/// process. Our integer assignment makes it so we don't have to compare huge strings.
///
/// This algorithm can be easily modified to compute the subset relation, so both are implemented
/// within the single [`Self::eq_aux`]. See also [`Set::from_mset`](crate::Set::from_mset), which
/// uses similar ideas.
impl PartialEq for Mset {
    fn eq(&self, other: &Self) -> bool {
        self.eq_aux(other, true)
    }
}

impl Mset {
    /// The empty set Ø.
    pub const fn empty() -> Self {
        Self(Vec::new())
    }

    /// Returns whether the multiset is finite.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Set cardinality.
    pub fn card(&self) -> usize {
        self.0.len()
    }

    /// An iterator over the elements of the [`Mset`].
    pub fn iter(&self) -> std::slice::Iter<Mset> {
        self.0.iter()
    }

    /// A mutable iterator over the elements of the [`Mset`].
    pub fn iter_mut(&mut self) -> std::slice::IterMut<Mset> {
        self.0.iter_mut()
    }

    /// Finds the [`Ahu`] encoding for a multiset.
    pub fn ahu(&self) -> Ahu {
        Ahu::new(self)
    }

    /// Set membership ∈.
    pub fn mem(&self, other: &Self) -> bool {
        self.iter().find(|&set| set == other).is_some()
    }

    /// Subset ⊆.
    pub fn subset(&self, other: &Self) -> bool {
        self.eq_aux(other, false)
    }

    /// Mutable set insertion.
    pub fn insert_mut(&mut self, other: Self) {
        self.0.push(other);
    }

    /// Set insertion x ∪ {y}.
    #[must_use]
    pub fn insert(mut self, other: Self) -> Self {
        self.insert_mut(other);
        self
    }

    /// Set singleton {x}.
    pub fn singleton(self) -> Self {
        Self(vec![self])
    }

    /// Set pair {x, y}.
    pub fn pair(self, other: Self) -> Self {
        Self(vec![self, other])
    }

    /// Set union x ∪ y.
    pub fn union(mut self, other: Self) -> Self {
        self.0.extend(other);
        self
    }

    /// Set union ∪x.
    pub fn big_union(self) -> Self {
        self.into_iter().flatten().collect()
    }

    /// Mutable set specification.
    pub fn select_mut<P: FnMut(&Mset) -> bool>(&mut self, mut pred: P) {
        let mut i = 0;
        while i < self.card() {
            if pred(&self.0[i]) {
                i += 1;
            } else {
                self.0.swap_remove(i);
            }
        }
    }

    /// Set specification.
    pub fn select<P: FnMut(&Mset) -> bool>(mut self, pred: P) -> Self {
        self.select_mut(pred);
        self
    }

    /// Powerset 2^x.
    pub fn powerset(self) -> Self {
        let n = self.card();
        let mut powerset = Self(vec![Self::empty()]);
        if n == 0 {
            return powerset;
        }

        for mut i in 1..((1 << n) - 1) {
            let mut subset = Mset::empty();
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

    /// The von Neumann rank of the set.
    pub fn rank(&self) -> usize {
        if let Some(max) = self.iter().map(Mset::rank).max() {
            max + 1
        } else {
            0
        }
    }

    /// The von Neumann ordinal for n.
    pub fn nat(n: usize) -> Self {
        let mut res = Vec::new();
        for _ in 0..n {
            res.push(Self(res.clone()));
        }
        Self(res)
    }

    /// The von Neumann hierarchy.
    pub fn neumann(n: usize) -> Self {
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
}
