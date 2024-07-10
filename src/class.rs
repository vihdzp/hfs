//! [`Classes`](Class) of hereditarily finite [`Sets`](Set).

use crate::prelude::*;

// -------------------- Iterators -------------------- //

/// Deduplicates an iterator over sets.
///
/// Note that this requires keeping a copy of all previous unique outputs to compare to.
#[derive(Clone)]
pub struct Dedup<I: Iterator<Item = Set>> {
    /// The iterator to deduplicate.
    iter: I,
    /// All previous unique outputs.
    output: Set,
}

impl<I: Iterator<Item = Set>> Dedup<I> {
    /// Deduplicates an iterator over sets.
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            output: Set::empty(),
        }
    }
}

impl<I: Iterator<Item = Set>> Iterator for Dedup<I> {
    type Item = Set;
    fn next(&mut self) -> Option<Self::Item> {
        for set in self.iter.by_ref() {
            if !self.output.contains(&set) {
                // Safety: we just performed the relevant check.
                unsafe {
                    self.output.insert_mut_unchecked(set.clone());
                }
                return Some(set);
            }
        }

        None
    }
}

/// Interleaves multiple iterators, getting all the elements from each.
///
/// We make no guarantee on the order in which elements are returned, other than the fact that each
/// iterator will be called until it returns `None`, even in the presence of infinite iterators.
#[derive(Clone, Default)]
pub struct Interleave<I: Iterator> {
    /// The iterators to interleave.
    iters: Vec<I>,
    /// The element from which we get the next iterator.
    index: usize,
}

impl<I: Iterator> Interleave<I> {
    /// Interleaves a set of iterators.
    #[must_use]
    pub const fn new(iters: Vec<I>) -> Self {
        Self { iters, index: 0 }
    }
}

impl<I: Iterator> Iterator for Interleave<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<I::Item> {
        // Attempts to get an element from each iterator.
        while !self.iters.is_empty() {
            // Wrap index around.
            debug_assert!(self.index <= self.iters.len());
            if self.index >= self.iters.len() {
                self.index = 0;
            }

            let next = self.iters[self.index].next();
            if next.is_some() {
                // By increasing the index, we guarantee we get elements out of every iterator.
                self.index += 1;
                return next;
            }

            // Remove spent iterator.
            self.iters.swap_remove(self.index);
        }

        None
    }
}

/// Outputs all pairs of elements from two iterators.
///
/// We make no guarantee on the order in which elements are returned, other than the fact that each
/// iterator will be called until it returns `None`, even in the presence of infinite iterators.
pub struct Product<I: Iterator, J: Iterator> {
    /// The first iterator.
    fst: I,
    /// The second iterator
    snd: J,
    /// The list of previous values from the first iterator.
    fst_values: Vec<I::Item>,
    /// The list of previous values from the second iterator.
    snd_values: Vec<J::Item>,

    /// A value determining which elements to grab for the next returned pair.
    index: usize,
}

impl<I: Iterator, J: Iterator> Product<I, J>
where
    I::Item: Clone,
    J::Item: Clone,
{
    /// Initializes a new iterator over pairs.
    pub const fn new(fst: I, snd: J) -> Self {
        Self {
            fst,
            snd,
            fst_values: Vec::new(),
            snd_values: Vec::new(),
            index: 0,
        }
    }
}

impl<I: Iterator, J: Iterator> Iterator for Product<I, J>
where
    I::Item: Clone,
    J::Item: Clone,
{
    type Item = (I::Item, J::Item);

    fn next(&mut self) -> Option<Self::Item> {
        // Despite using auxiliary variable names for compactness, we don't mutate these arrays and
        // the index until we're sure we won't return `None`. In that way, even if our iterators are
        // buggy and return `Some(x)` after returning `None`, the overall logic is still sound.
        let a = self.fst_values.len();
        let b = self.snd_values.len();
        let idx = self.index;

        // We're adding elements where the first entry comes later in its iterator.
        Some(if a > b {
            // We added all we could, switch mode.
            if idx >= b {
                debug_assert_eq!(idx, b);
                if let Some(snd_value) = self.snd.next() {
                    self.snd_values.push(snd_value.clone());
                    self.index = 1;
                    (self.fst_values[0].clone(), snd_value)
                }
                // The second iterator ran out.
                else {
                    let fst_value = self.fst.next()?;
                    self.fst_values.push(fst_value.clone());
                    self.index = 1;
                    // Safety: We could only have added an element to the first list if we also
                    // added one to the second list.
                    unsafe {
                        (
                            fst_value,
                            self.snd_values.first().unwrap_unchecked().clone(),
                        )
                    }
                }
            } else {
                // Add next element.
                self.index += 1;
                (
                    self.fst_values.last().unwrap().clone(),
                    self.snd_values[idx].clone(),
                )
            }
        }
        // We're adding elements where the second entry comes later in its iterator.
        else {
            // We added all we could, switch mode.
            if idx >= a {
                debug_assert_eq!(idx, a);
                if let Some(fst_value) = self.fst.next() {
                    if let Some(snd_value) = self.snd_values.first() {
                        self.fst_values.push(fst_value.clone());
                        self.index = 1;
                        (fst_value, snd_value.clone())
                    }
                    // Special case: the very first value returned by the iterator.
                    else {
                        let snd_value = self.snd.next()?;
                        self.fst_values.push(fst_value.clone());
                        self.snd_values.push(snd_value.clone());
                        self.index = 1;
                        (fst_value, snd_value)
                    }
                }
                // The first iterator ran out.
                else {
                    let fst_value = self.fst_values.first()?;
                    let snd_value = self.snd.next()?;
                    self.snd_values.push(snd_value.clone());
                    self.index = 1;
                    (fst_value.clone(), snd_value)
                }
            } else {
                // Add next element.
                self.index += 1;
                (
                    self.fst_values[idx].clone(),
                    self.snd_values.last().unwrap().clone(),
                )
            }
        })
    }
}

/// Shorthand for defining [`Nat`] and [`Zermelo`].
macro_rules! naturals {
    ($class: ident, $func: ident, $name: literal) => {
        #[doc = concat!("The class of ", $name, " naturals ℕ.")]
        #[derive(Clone, Default)]
        pub struct $class(usize);

        impl $class {
            #[doc = concat!("Initializes an iterator over  ", $name, " naturals.")]
            #[must_use]
            pub const fn new() -> Self {
                Self(0)
            }
        }

        impl Iterator for $class {
            type Item = Set;

            fn next(&mut self) -> Option<Set> {
                let res = Set::$func(self.0);
                self.0 += 1;
                Some(res)
            }

            fn nth(&mut self, n: usize) -> Option<Set> {
                self.0 += n;
                self.next()
            }
        }
    };
}

naturals!(Nat, nat, "von Neumann");
naturals!(Zermelo, zermelo, "Zermelo");

/// The universal class.
#[derive(Clone, Default)]
pub struct Univ(Vec<Set>);

impl Univ {
    /// Initializes the universal class.
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }
}

impl Iterator for Univ {
    type Item = Set;
    fn next(&mut self) -> Option<Set> {
        // The set m is contained in the set n iff n.testbit(m).
        let mut set = Set::empty();
        let mut n = self.0.len();

        let mut m = 0;
        while n != 0 {
            if n % 2 == 1 {
                // Safety: all sets we build are distinct.
                unsafe {
                    set.insert_mut_unchecked(self.0.get_unchecked(m).clone());
                }
            }

            n /= 2;
            m += 1;
        }

        // In theory, this procedure continued forever would build every single hereditarily finite
        // set, exactly once. Practically, we can't build more than 2^32 or 2^64 sets before our
        // allocation overflows, and even the latter number is overkill. Because of this, we opt to
        // save space instead of pretending mathematical correctness.
        if self.0.len() < usize::BITS as usize {
            self.0.push(set.clone());
        }
        Some(set)
    }
}

// -------------------- Classes -------------------- //

/// A [class](https://en.wikipedia.org/wiki/Class_(set_theory)) of hereditarily finite sets.
///
/// We define this as an arbitrary iterator over [`Set`] without duplicate entries. This allows for
/// many important constructions, such as the [universal class](Self::univ) or for [classes from a
/// predicate](Self::pred).
///
/// Note that since classes represent potentially infinite collections, it's possible for a piece of
/// code to hang. For instance,
///
/// ```rust,no_run
/// # use hfs::prelude::Class;
/// Class::pred(|_| false).into_iter().next();
/// ```
///
/// will run forever, unsucessfully testing the predicate on each set. This limitation means that
/// certain relations, like membership or subsets, cannot be defined.
///
/// ## Invariants
///
/// Every two elements in a [`Class`] must be distinct. **Unsafe code performs optimizations
/// contingent on this.**
#[derive(IntoIterator)]
pub struct Class(Box<dyn Iterator<Item = Set>>);

impl Default for Class {
    fn default() -> Self {
        Self::empty()
    }
}

impl From<Set> for Class {
    fn from(value: Set) -> Self {
        Self::new(value.into_iter())
    }
}

impl Class {
    /// Initializes a new class from an iterator.
    ///
    /// ## Safety
    ///
    /// You must guarantee that any two sets returned by the iterator are distinct.
    pub unsafe fn new_unchecked<I: Iterator<Item = Set> + 'static>(iter: I) -> Self {
        Self(Box::new(iter))
    }

    /// Initializes a new class from an iterator. This iterator is deduplicated via [`Dedup`].
    pub fn new<I: Iterator<Item = Set> + 'static>(iter: I) -> Self {
        // Safety: we are deduplicating the iteraor.
        unsafe { Self::new_unchecked(Dedup::new(iter)) }
    }

    /// The empty class Ø.
    #[must_use]
    pub fn empty() -> Self {
        // Safety: this iterator has no duplicates.
        unsafe { Self::new_unchecked(std::iter::empty()) }
    }

    /// The universal class.
    #[must_use]
    pub fn univ() -> Self {
        // Safety: this iterator has no duplicates.
        unsafe { Self::new_unchecked(Univ::new()) }
    }

    /// The class of naturals ℕ.
    #[must_use]
    pub fn nat() -> Self {
        // Safety: this iterator has no duplicates.
        unsafe { Self::new_unchecked(Nat::new()) }
    }

    /// The class of Zermelo naturals ℕ.
    #[must_use]
    pub fn zermelo() -> Self {
        // Safety: this iterator has no duplicates.
        unsafe { Self::new_unchecked(Zermelo::new()) }
    }

    /// Class specification.
    #[must_use]
    pub fn select<P: FnMut(&Set) -> bool + 'static>(self, pred: P) -> Self {
        // Safety: if the original class has no duplicates, neither does this one.
        unsafe { Self::new_unchecked(self.into_iter().filter(pred)) }
    }

    /// Initializes a new class from a predicate.
    ///
    /// This iterates over the universal class and selects the elements satisfying the predicate. If
    /// they ever run out, the iterator will continue searching forever.
    pub fn pred<P: FnMut(&Set) -> bool + 'static>(pred: P) -> Self {
        Self::univ().select(pred)
    }

    /// Class union over a vector.
    ///
    /// This internally uses [`Interleave`] to get the elements out of each set.
    pub fn union_vec(vec: Vec<Self>) -> Self {
        Self::new(Interleave::new(
            vec.into_iter().map(IntoIterator::into_iter).collect(),
        ))
    }

    /// Class union.
    ///
    /// This internally uses [`Interleave`] to get the elements out of each set.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self::union_vec(vec![self, other])
    }

    /// Cartesian product of classes.
    ///
    /// This internally uses [`Product`] to get the elements out of each set.
    #[must_use]
    pub fn prod(self, other: Self) -> Self {
        // Safety: the pairs of sets returned by `Product` are unique, and unique pairs are mapped
        // to unique sets by `kpair`.
        unsafe {
            Self::new_unchecked(
                Product::new(self.into_iter(), other.into_iter()).map(|(x, y)| x.kpair(y)),
            )
        }
    }

    /// [Replaces](https://en.wikipedia.org/wiki/Axiom_schema_of_replacement) the elements in a
    /// class by applying a function.
    #[must_use]
    pub fn replacement<F: FnMut(Set) -> Set + 'static>(self, func: F) -> Self {
        Self::new(self.into_iter().map(func))
    }

    /// [Replaces](https://en.wikipedia.org/wiki/Axiom_schema_of_replacement) the elements in a
    /// class by applying a function. Does not check that the outputs are all distinct.
    ///
    /// ## Safety
    ///
    /// You must guarantee that the function does not yield the same output for two distinct
    /// elements of the class.
    #[must_use]
    pub unsafe fn replacement_unchecked<F: FnMut(Set) -> Set + 'static>(self, func: F) -> Self {
        Self::new_unchecked(self.into_iter().map(func))
    }

    /// [Chooses](https://en.wikipedia.org/wiki/Axiom_of_choice) an arbitrary element from a
    /// non-empty class.
    pub fn choose(&mut self) -> Option<Set> {
        self.0.next()
    }
}
