//! # Hereditarily finite sets

#![warn(clippy::pedantic)]
#![warn(missing_docs)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_docs_in_private_items)]

pub mod mset;
pub mod prelude;
pub mod set;
pub mod utils;

use prelude::*;

/// Small vector.
type SmallVec<T> = smallvec::SmallVec<[T; 4]>;

/// [`smallvec::smallvec`] coerced into [`SmallVec`].
#[macro_export]
macro_rules! smallvec {
    ($elem: expr; $n: expr) => (
        SmallVec::from_elem($elem, $n)
    );
    ($($x: expr), *$(,)*) => ({
        let vec: SmallVec<_> = smallvec::smallvec![$($x,)*];
        vec
    });
}

/// Transmute a vector of one type into a vector of another type.
///
/// ## Safety
///
/// The types `T` and `U` must be transmutable into each other. In particular, they must have the
/// same size and alignment.
#[allow(dead_code)]
unsafe fn transmute_vec<T, U>(vec: Vec<T>) -> Vec<U> {
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());

    let mut vec = std::mem::ManuallyDrop::new(vec);
    unsafe { Vec::from_raw_parts(vec.as_mut_ptr().cast(), vec.len(), vec.capacity()) }
}

/// A seal for [`SetTrait`], avoiding foreign implementations.
trait Seal {}

/// A trait for [`Mset`] and [`Set`].
///
/// The trait is sealed so that these are the only two types that ever implement it.
#[allow(private_bounds)]
pub trait SetTrait:
    Seal
    + AsRef<Mset>
    + Clone
    + Debug
    + Default
    + Display
    + Eq
    + Into<Vec<Self>>
    + IntoIterator<Item = Self>
    + PartialOrd
{
    // -------------------- Basic methods -------------------- //

    /// The set as a slice.
    fn as_slice(&self) -> &[Self];

    /// Removes all elements from the set.
    fn clear(&mut self);

    /// Set cardinality.
    fn card(&self) -> usize {
        self.as_slice().len()
    }

    /// Whether the set is empty.
    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Iterate over the elements of the set.
    fn iter(&self) -> std::slice::Iter<Self> {
        self.as_slice().iter()
    }

    /// Get the [`Ahu`] encoding for the set.
    fn ahu(&self) -> Ahu {
        Ahu::new(self.as_ref())
    }

    /// Von Neumann set rank.
    fn rank(&self) -> usize {
        Levels::init(self.as_ref()).fill().rank()
    }

    // -------------------- Constructions -------------------- //

    /// Empty set Ø.
    fn empty() -> Self;

    /// Singleton set {x}.
    #[must_use]
    fn singleton(self) -> Self;

    /// In-place set insertion x ∪ {y}.
    fn insert_mut(&mut self, set: Self);

    /// Set insertion x ∪ {y}.
    #[must_use]
    fn insert(mut self, set: Self) -> Self {
        self.insert_mut(set);
        self
    }

    /// In-place set specification.
    fn select_mut<P: FnMut(&Self) -> bool>(&mut self, pred: P);

    /// Set specification.
    #[must_use]
    fn select<P: FnMut(&Self) -> bool>(mut self, pred: P) -> Self {
        self.select_mut(pred);
        self
    }

    /// Set pair {x, y}.
    #[must_use]
    fn pair(self, other: Self) -> Self {
        self.singleton().insert(other)
    }

    /// Union x ∪ y.
    #[must_use]
    fn union(self, other: Self) -> Self;

    /// Union over an iterator.
    fn union_iter<I: IntoIterator<Item = Self>>(iter: I) -> Self;

    /// Union ∪x.
    #[must_use]
    fn big_union(self) -> Self {
        Self::union_iter(self)
    }

    /// Powerset P(x).
    #[must_use]
    fn powerset(self) -> Self;

    /// The von Neumann encoding for a natural.
    fn nat(n: usize) -> Self;

    /// The Zermelo encoding for a natural.
    fn zermelo(n: usize) -> Self;

    /// The von Neumann hierarchy.
    fn neumann(n: usize) -> Self;

    // -------------------- Relations -------------------- //

    /// **Internal method.**
    ///
    /// Determines whether `fst` is a subset of `snd`. Is optimized separately when the levels
    /// correspond to a set or a multiset.
    ///
    /// See [`Levels::subset_gen`].
    ///
    /// ## Safety
    ///
    /// Each of the levels must have been validly built from a set of type `Self`.
    unsafe fn _levels_subset(fst: &Levels<&Mset>, snd: &Levels<&Mset>) -> bool;

    /// Subset relation ⊆.
    fn subset(&self, other: &Self) -> bool {
        self.le(other)
    }

    /// Strict subset relation ⊂.
    fn ssubset(&self, other: &Self) -> bool {
        self.lt(other)
    }

    /// Membership relation ∈.
    fn contains(&self, other: &Self) -> bool {
        let mut fst = unsafe { Levels::empty() };
        let snd = Levels::init(other.as_ref()).fill();
        let mut buf = Vec::new();

        // Check equality between every set in `self` and `other`.
        self.iter().any(move |set| {
            // `fst` must have exactly as many levels as `snd` of the same lengths.
            fst.init_mut(set.as_ref());
            while fst.step(&mut buf) {
                if let Some(level) = snd.get(fst.last_idx()) {
                    if fst.last().len() != level.len() {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            fst.level_len() == snd.level_len() && unsafe { Self::_levels_subset(&fst, &snd) }
        })
    }
}

/// Implements [`PartialOrd`] for [`SetTrait`].
macro_rules! impl_partial_ord {
    ($t: ty) => {
        impl PartialEq for $t {
            fn eq(&self, other: &Self) -> bool {
                if let Some((fst, snd)) = Levels::eq_levels(self.as_ref(), other.as_ref()) {
                    unsafe { <$t>::_levels_subset(&fst, &snd) }
                } else {
                    false
                }
            }
        }

        impl PartialOrd for $t {
            fn le(&self, other: &Self) -> bool {
                if let Some((fst, snd)) = Levels::le_levels(self.as_ref(), other.as_ref()) {
                    unsafe { <$t>::_levels_subset(&fst, &snd) }
                } else {
                    false
                }
            }

            fn ge(&self, other: &Self) -> bool {
                other.le(self)
            }

            fn lt(&self, other: &Self) -> bool {
                self.card() < other.card() && self.le(other)
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
    };
}

impl_partial_ord!(Mset);
impl_partial_ord!(Set);
