//! # silicon-kit
//!
//! Safe, zero-overhead hardware silicon intrinsics, CPU feature capability tokens,
//! streaming stores, F16C, and bit manipulation for Rust.
//!
//! Designed with the Scrooge Systems Mindset: `#![no_std]`, zero heap allocations,
//! and complete compile-time inlining to single CPU instructions without wrapper overhead.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod caps;
pub use caps::CpuCaps;

#[cfg(feature = "f16c")]
pub mod f16c;

#[cfg(feature = "streaming")]
pub mod streaming;

#[cfg(feature = "bits")]
pub mod bits;

#[cfg(feature = "snorm")]
pub mod snorm;

#[cfg(feature = "crc32")]
pub mod crc32;
