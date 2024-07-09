//! Crate prelude.

// The actual prelude.
pub use crate::{
    class::{self, Class},
    mset::{Mset, SetError},
    set::Set,
    utils::Ahu,
    SetTrait,
};

// Convenient imports within the crate.
pub(crate) use crate::{
    has_consecutive,
    utils::{btree_index, Levels},
    SmallVec,
};
pub(crate) use bitvec::prelude::*;
pub(crate) use derive_more::IntoIterator;
pub(crate) use std::{
    cmp::Ordering,
    collections::{btree_map::Entry, BTreeMap},
    fmt::{Debug, Display, Formatter, Result as FmtResult, Write},
    hint, mem, slice,
    str::FromStr,
};
