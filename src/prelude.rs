//! Crate prelude.

// The actual prelude.
pub use crate::{
    class::{self, Class},
    mset::{Mset, SetError},
    set::Set,
    levels::Ahu,
    SetTrait,
};

// Convenient imports within the crate.
pub(crate) use crate::{
    consecutive_eq,
    levels::{btree_index, Compare, Levels},
    SmallVec,
};
pub(crate) use bitvec::prelude::*;
pub(crate) use derive_more::{AsRef, Display, Into, IntoIterator};
pub(crate) use std::{
    cmp::Ordering,
    collections::{btree_map::Entry, BTreeMap},
    fmt::{Debug, Display, Formatter, Result as FmtResult, Write},
    hint, mem, ptr, slice,
    str::FromStr,
};
