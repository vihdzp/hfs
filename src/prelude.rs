//! Crate prelude.

// The actual prelude.
pub use crate::{mset::Mset, set::Set, utils::Ahu};

// Convenient imports within the crate.
pub(crate) use crate::{utils::Levels, SmallVec};
pub(crate) use bitvec::prelude::*;
pub(crate) use derive_more::IntoIterator;
pub(crate) use std::{
    collections::{btree_map::Entry, BTreeMap},
    error::Error,
    fmt::{Debug, Display, Formatter, Result as FmtResult, Write},
    str::FromStr,
};
