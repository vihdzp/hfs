//! # Hereditarily finite sets

#![warn(clippy::pedantic)]
#![warn(missing_docs)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_docs_in_private_items)]
//#![warn(clippy::undocumented_unsafe_blocks)]

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

/// Whether a slice has consecutive elements.
fn has_consecutive<T: PartialEq>(slice: &[T]) -> bool {
    (1..slice.len()).any(|i| slice[i - 1] == slice[i])
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
    + FromStr<Err = SetError>
    + Into<Vec<Self>>
    + IntoIterator<Item = Self>
    + PartialOrd
{
    // -------------------- Basic methods -------------------- //

    /// The set as a slice.
    fn as_slice(&self) -> &[Self];

    /// **Internal method.**
    ///
    /// The set as a mutable slice.
    #[allow(clippy::missing_safety_doc)]
    unsafe fn _as_slice_mut(&mut self) -> &mut [Self];

    /// A reference to the inner vector.
    ///
    /// Note that internally, both kinds of set store [`Mset`].
    fn as_vec(&self) -> &Vec<Mset>;

    /// **Internal method.**
    ///
    /// A mutable reference to the inner vector.
    ///
    /// Note that internally, both kinds of set store [`Mset`].
    #[allow(clippy::missing_safety_doc)]
    unsafe fn _as_vec_mut(&mut self) -> &mut Vec<Mset>;

    /// Removes all elements from the set.
    fn clear(&mut self) {
        // Safety: The empty set is valid for both types.
        unsafe { self._as_vec_mut() }.clear();
    }

    /// Set cardinality.
    fn card(&self) -> usize {
        self.as_slice().len()
    }

    /// Whether the set is empty.
    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// The capacity of the backing vector.
    fn capacity(&self) -> usize {
        self.as_vec().capacity()
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

    /// In-place set insertion x + {y}.
    fn insert_mut(&mut self, set: Self);

    /// Set insertion x + {y}.
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

    /// Sum over a vector.
    ///
    /// See [`SetTrait::sum`].
    fn sum_vec(vec: Vec<Self>) -> Self;

    /// Sum x + y.
    ///
    /// - The sum of two multisets is the multiset created by directly appending the elements of
    ///   both.
    /// - The sum of two sets coincides with their union.
    #[must_use]
    fn sum(self, other: Self) -> Self {
        Self::sum_vec(vec![self, other])
    }

    /// Sum Σx.
    ///
    /// See [`SetTrait::sum`].
    #[must_use]
    fn big_sum(self) -> Self {
        Self::sum_vec(self.into())
    }

    /// Union over a vector.
    fn union_vec(vec: Vec<Self>) -> Self;

    /// Union x ∪ y.
    #[must_use]
    fn union(self, other: Self) -> Self {
        Self::union_vec(vec![self, other])
    }

    /// Union ∪x.
    #[must_use]
    fn big_union(self) -> Self {
        Self::union_vec(self.into())
    }

    /// Intersection over a vector.
    ///
    /// The intersection of an empty family would be the universal set, which can't be returned.
    fn inter_vec(vec: Vec<Self>) -> Option<Self>;

    /// Intersection x ∩ y.
    #[must_use]
    fn inter(self, other: Self) -> Self {
        // Safety: 2 != 0.
        unsafe { Self::inter_vec(vec![self, other]).unwrap_unchecked() }
    }

    /// Intersection ∩x.
    ///
    /// The intersection of an empty family would be the universal set, which can't be returned.
    #[must_use]
    fn big_inter(self) -> Option<Self> {
        Self::inter_vec(self.into())
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
    /// See [`Levels::both_ahu`].
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
        // Safety: this buffer is only used to initialize the first set in `self`.
        let mut fst = unsafe { Levels::empty() };
        let snd = Levels::init(other.as_ref()).fill();
        let mut buf = Vec::new();

        // Check equality between every set in `self` and `other`.
        self.iter().any(move |set| {
            // `fst` must have exactly as many levels as `snd` of the same lengths.
            fst.init_mut(set.as_ref());
            while fst.step(&mut buf) {
                if let Some(level) = snd.get(fst.rank()) {
                    if fst.last().len() != level.len() {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Safety: both `fst` and `snd` are valid for `Levels<&Self>`.
            fst.level_len() == snd.level_len() && unsafe { Self::_levels_subset(&fst, &snd) }
        })
    }

    /// Checks whether two sets are disjoint.
    fn disjoint(&self, other: &Self) -> bool {
        Self::disjoint_pairwise([self, other])
    }

    /// Checks whether a list of sets are disjoint.
    ///
    /// For pairwise disjoint sets, see [`SetTrait::disjoint_pairwise`].
    fn disjoint_iter<'a, I: IntoIterator<Item = &'a Self>>(iter: I) -> bool
    where
        Self: 'a;

    /// Checks whether a list of sets are pairwise disjoint.
    ///
    /// For non-pairwise disjoint sets, see [`SetTrait::disjoint_iter`].
    fn disjoint_pairwise<'a, I: IntoIterator<Item = &'a Self>>(iter: I) -> bool
    where
        Self: 'a;

    // -------------------- Internals -------------------- //

    /// **Internal method.**
    ///
    ///  Normalize string for a set.
    #[must_use]
    fn _normalize(str: &str) -> String {
        let set: Self = str.parse().unwrap();
        set.to_string()
    }

    /// **Internal method.**
    ///
    /// Verify round-trip conversion between a set and a string.
    fn _roundtrip(&self, str: &str) {
        assert_eq!(self, &str.parse().unwrap());
        assert_eq!(self.to_string(), str);
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
