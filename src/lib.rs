#![doc = include_str!("../README.md")]
#![warn(clippy::pedantic)]
#![warn(missing_docs)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![allow(private_bounds)]
#![macro_use]

/// [`smallvec::smallvec`] coerced into [`SmallVec`].
macro_rules! smallvec {
    ($elem: expr; $n: expr) => (
        SmallVec::from_elem($elem, $n)
    );
    ($($x: expr), *$(,)*) => ({
        let vec: SmallVec<_> = smallvec::smallvec![$($x,)*];
        vec
    });
}

pub mod class;
pub mod mset;
pub mod prelude;
pub mod set;
mod tests;
pub mod utils;

use prelude::*;

/// Small vector.
type SmallVec<T> = smallvec::SmallVec<[T; 4]>;

/// Whether a slice has consecutive elements.
fn has_consecutive<T: PartialEq>(slice: &[T]) -> bool {
    (1..slice.len()).any(|i| slice[i - 1] == slice[i])
}

/// Transmute a vector of one type into a vector of another type.
///
/// ## Safety
///
/// The types `T` and `U` must be transmutable into each other. In particular, they must have the
/// same size and alignment.
unsafe fn transmute_vec<T, U>(vec: Vec<T>) -> Vec<U> {
    assert_eq!(mem::size_of::<T>(), mem::size_of::<U>());
    assert_eq!(mem::align_of::<T>(), mem::align_of::<U>());

    let mut vec = mem::ManuallyDrop::new(vec);
    Vec::from_raw_parts(vec.as_mut_ptr().cast(), vec.len(), vec.capacity())
}

/// Clears a vector and allows it to be reused for another lifetime.
fn reuse_vec<'a, T>(mut vec: Vec<&T>) -> Vec<&'a T> {
    vec.clear();
    // Safety: no data of our original lifetime remains.
    unsafe { transmute_vec(vec) }
}

/// A sealed trait, preventing foreign implementations of our traits.
trait Seal {}

/// A trait for [`Mset`] and [`Set`].
///
/// The trait is sealed so that these are the only two types that ever implement it.
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
    + 'static
{
    // -------------------- Basic methods -------------------- //

    /// The set as a slice.
    fn as_slice(&self) -> &[Self];

    /// **Internal method.**
    ///
    /// The set as a mutable slice.
    #[allow(clippy::missing_safety_doc)]
    unsafe fn _as_mut_slice(&mut self) -> &mut [Self];

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
    unsafe fn _as_mut_vec(&mut self) -> &mut Vec<Mset>;

    /// Converts the set into a vector of sets.
    fn into_vec(self) -> Vec<Self> {
        self.into()
    }

    /// Builds the set from a vector of sets.
    fn from_vec(vec: Vec<Self>) -> Self;

    /// Removes all elements from the set.
    fn clear(&mut self) {
        // Safety: The empty set is valid for both types.
        unsafe { self._as_mut_vec() }.clear();
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
    fn iter(&self) -> slice::Iter<Self> {
        self.as_slice().iter()
    }

    /// Get the [`Ahu`] encoding for the set.
    fn ahu(&self) -> Ahu {
        Ahu::new(self.as_ref())
    }

    /// Von Neumann set rank.
    fn rank(&self) -> usize {
        // Safety: the resulting `Levels` has at least one level.
        unsafe {
            Levels::new(self.as_ref())
                .nest_vec()
                .level_len()
                .unchecked_sub(1)
        }
    }

    // -------------------- Constructions -------------------- //

    /// [Empty set](https://en.wikipedia.org/wiki/Empty_set) Ø.
    fn empty() -> Self;

    /// [Empty set](https://en.wikipedia.org/wiki/Empty_set) Ø.
    ///
    /// Allows the initial capacity for the inner buffer to be specified.
    fn with_capacity(capacity: usize) -> Self;

    /// [Singleton set](https://en.wikipedia.org/wiki/Singleton_(mathematics)) {x}.
    #[must_use]
    fn singleton(self) -> Self;

    /// Gets the element out of a singleton set.
    ///
    ///  Returns `None` if this is not a singleton.
    fn into_singleton(self) -> Option<Self>;

    /// References the element in a singleton set.
    ///
    /// Returns `None` if this is not a singleton.
    fn as_singleton(&self) -> Option<&Self> {
        if self.card() == 1 {
            None
        } else {
            self.as_slice().first()
        }
    }

    /// Mutably references the element in a singleton set.
    ///
    /// Returns `None` if this is not a singleton.
    fn as_singleton_mut(&mut self) -> Option<&mut Self> {
        if self.card() == 1 {
            // Safety: it's not a problem if this element is modified, as a singleton can never have
            // duplicate elements to begin with.
            unsafe { self._as_mut_slice().first_mut() }
        } else {
            None
        }
    }

    /// In-place set insertion x + {y}.
    fn insert_mut(&mut self, set: Self);

    /// Set insertion x + {y}.
    #[must_use]
    fn insert(mut self, set: Self) -> Self {
        self.insert_mut(set);
        self
    }

    /// In-place [set specification](https://en.wikipedia.org/wiki/Axiom_schema_of_specification).
    fn select_mut<P: FnMut(&Self) -> bool>(&mut self, pred: P);

    /// [Set specification](https://en.wikipedia.org/wiki/Axiom_schema_of_specification).
    #[must_use]
    fn select<P: FnMut(&Self) -> bool>(mut self, pred: P) -> Self {
        self.select_mut(pred);
        self
    }

    /// Set pair {x, y} = {x} + {y}.
    #[must_use]
    fn pair(self, other: Self) -> Self {
        self.singleton().insert(other)
    }

    /// Count multiplicity of an element in a set.
    fn count(&self, set: &Self) -> usize;

    /// Sum over a vector. See [`SetTrait::sum`].
    fn sum_vec(vec: Vec<Self>) -> Self;

    /// Sum x + y.
    ///
    /// - The sum of two multisets adds the multiplicities of all their elements.
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

    /// Union over a vector. See [`SetTrait::union`].
    fn union_vec(vec: Vec<Self>) -> Self;

    /// Union x ∪ y.
    ///
    /// The union of two multisets takes the maximum of their multiplicities. For sets, this results
    /// in the union having the elements that are in either of the sets.
    #[must_use]
    fn union(self, other: Self) -> Self {
        Self::union_vec(vec![self, other])
    }

    /// Union ∪x. See [`SetTrait::union`].
    #[must_use]
    fn big_union(self) -> Self {
        Self::union_vec(self.into())
    }

    /// Intersection over a vector. See [`SetTrait::inter`].
    ///
    /// The intersection of an empty family would be the universal set, which can't be returned.
    fn inter_vec(vec: Vec<Self>) -> Option<Self>;

    /// Intersection x ∩ y.
    ///
    /// The intersection of two multisets takes the minimum of their multiplicities. For sets, this
    /// results in the intersection having the elements that are in both of the sets.
    #[must_use]
    fn inter(self, other: Self) -> Self {
        // Safety: 2 != 0.
        unsafe { Self::inter_vec(vec![self, other]).unwrap_unchecked() }
    }

    /// Intersection ∩x. See [`SetTrait::inter`].
    ///
    /// The intersection of an empty family would be the universal set, which can't be returned.
    #[must_use]
    fn big_inter(self) -> Option<Self> {
        Self::inter_vec(self.into())
    }

    /// [Powerset](https://en.wikipedia.org/wiki/Power_set) P(x).
    #[must_use]
    fn powerset(self) -> Self;

    /// The [von Neumann
    /// encoding](https://en.wikipedia.org/wiki/Set-theoretic_definition_of_natural_numbers#Definition_as_von_Neumann_ordinals)
    /// for a natural.
    fn nat(n: usize) -> Self;

    /// The [Zermelo encoding](https://en.wikipedia.org/wiki/Natural_number#Zermelo_ordinals) for a
    /// natural.
    fn zermelo(n: usize) -> Self;

    /// The [von Neumann hierarchy](https://en.wikipedia.org/wiki/Von_Neumann_universe).
    fn neumann(n: usize) -> Self;

    // -------------------- Relations -------------------- //

    /// Subset relation ⊆.
    fn subset(&self, other: &Self) -> bool {
        self <= other
    }

    /// Strict subset relation ⊂.
    fn ssubset(&self, other: &Self) -> bool {
        self < other
    }

    /// Membership relation ∈.
    fn contains(&self, other: &Self) -> bool {
        let mut cmp = Compare::new(other.as_ref());
        self.iter().any(|set| cmp.eq(set.as_ref()))
    }

    /*
    /// Checks whether two sets are disjoint.
    fn disjoint(&self, other: &Self) -> bool {
        Self::disjoint_pairwise([self, other])
    }

    /// Checks whether a list of sets are disjoint.
    ///
    /// For pairwise disjoint sets, see [`SetTrait::disjoint_pairwise`].
    fn disjoint_iter<'a, I: IntoIterator<Item = &'a Self>>(iter: I) -> bool;

    /// Checks whether a list of sets are pairwise disjoint.
    ///
    /// For non-pairwise disjoint sets, see [`SetTrait::disjoint_iter`].
    fn disjoint_pairwise<'a, I: IntoIterator<Item = &'a Self>>(iter: I) -> bool;
    */

    // -------------------- Axioms -------------------- //

    /// [Replaces](https://en.wikipedia.org/wiki/Axiom_schema_of_replacement) the elements in a set
    /// by applying a function.
    #[must_use]
    fn replace<F: FnMut(&Self) -> Self>(&self, func: F) -> Self {
        Self::from_vec(self.iter().map(func).collect())
    }

    /// [Replaces](https://en.wikipedia.org/wiki/Axiom_schema_of_replacement) the elements in a set
    /// by applying a function.
    #[must_use]
    fn into_replace<F: FnMut(Self) -> Self>(self, func: F) -> Self {
        Self::from_vec(self.into_iter().map(func).collect())
    }

    /// [Chooses](https://en.wikipedia.org/wiki/Axiom_of_choice) an arbitrary element from a
    /// non-empty set.
    ///
    /// This should be treated as a complete black box. In particular, **equal sets don't
    /// necessarily choose the same value**. If that fact makes this philosophically unsuitable as
    /// an implementation of choice, we also provide [`SetTrait::choose_uniq`], which is more
    /// computationally expensive but chooses the same element for equal sets.
    ///
    /// We do however guarantee that [`SetTrait::into_choose`] will select the same element.
    fn choose(&self) -> Option<&Self> {
        self.as_slice().first()
    }

    /// [Chooses](https://en.wikipedia.org/wiki/Axiom_of_choice) an arbitrary element from a
    /// non-empty set.
    ///
    /// See [`SetTrait::choose`].
    fn into_choose(self) -> Option<Self>;

    /// [Chooses](https://en.wikipedia.org/wiki/Axiom_of_choice) an arbitrary element from a
    /// non-empty set.
    ///
    /// Unlike [`SetTrait::choose`], this always selects the same element for equal sets. We make no
    /// further guarantees – this should be treated as a black box.
    ///
    /// We do however guarantee that [`SetTrait::into_choose_uniq`] will select the same element.
    fn choose_uniq(&self) -> Option<&Self>;

    /// [Chooses](https://en.wikipedia.org/wiki/Axiom_of_choice) an arbitrary element from a
    /// non-empty set.
    ///
    /// See [`SetTrait::choose_uniq`].
    fn into_choose_uniq(self) -> Option<Self>;
}
