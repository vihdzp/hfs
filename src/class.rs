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

    /// Initializes a new class from an iterator. This iterator is deduplicated.
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
}
