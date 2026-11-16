//! CPU Feature Discovery and Capabilities Token.
//!
//! Provides zero-overhead capability discovery. In release builds with target features
//! enabled at compile-time (e.g. `-C target-cpu=native`), all checks fold to `true` at compile time.

/// 64-bit bitmask representing detected hardware CPU capabilities.
///
/// Unforgeable capability token: fields are private to guarantee that capability
/// flags cannot be fabricated by safe code. To construct a valid token, query hardware
/// via [`CpuCaps::detect`], or use the safe zero-capability token [`CpuCaps::empty`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct CpuCaps {
    flags: u64,
}

impl CpuCaps {
    pub const AVX: u64 = 1 << 0;
    pub const AVX2: u64 = 1 << 1;
    pub const F16C: u64 = 1 << 2;
    pub const BMI1: u64 = 1 << 3;
    pub const BMI2: u64 = 1 << 4;
    pub const POPCNT: u64 = 1 << 5;
    pub const SSE42: u64 = 1 << 6;
    pub const AVX512F: u64 = 1 << 7;

    /// Return an empty capability token with no hardware features enabled.
    #[inline(always)]
    pub const fn empty() -> Self {
        Self { flags: 0 }
    }

    /// Construct a capability token from raw flags.
    ///
    /// # Safety
    /// The caller must ensure that the CPU actually supports all instruction set
    /// features indicated by `flags`. Falsely asserting capabilities will cause
    /// downstream intrinsic calls to execute invalid instructions, resulting in SIGILL crashes.
    #[inline(always)]
    pub const unsafe fn from_raw_flags(flags: u64) -> Self {
        Self { flags }
    }

    /// Retrieve the raw 64-bit capability bitmask.
    #[inline(always)]
    pub const fn flags(&self) -> u64 {
        self.flags
    }

    /// Detect current CPU capabilities.
    ///
    /// When standard library is enabled, queries runtime CPUID flags once.
    /// In `#![no_std]` without runtime detection, reflects compile-time target features.
    #[inline]
    pub fn detect() -> Self {
        #[allow(unused_mut)]
        let mut flags = 0u64;

        #[cfg(all(target_arch = "x86_64", feature = "std"))]
        {
            if std::is_x86_feature_detected!("avx") {
                flags |= Self::AVX;
            }
            if std::is_x86_feature_detected!("avx2") {
                flags |= Self::AVX2;
            }
            if std::is_x86_feature_detected!("f16c") {
                flags |= Self::F16C;
            }
            if std::is_x86_feature_detected!("bmi1") {
                flags |= Self::BMI1;
            }
            if std::is_x86_feature_detected!("bmi2") {
                flags |= Self::BMI2;
            }
            if std::is_x86_feature_detected!("popcnt") {
                flags |= Self::POPCNT;
            }
            if std::is_x86_feature_detected!("sse4.2") {
                flags |= Self::SSE42;
            }
            if std::is_x86_feature_detected!("avx512f") {
                flags |= Self::AVX512F;
            }
        }

        #[cfg(all(target_arch = "x86_64", not(feature = "std")))]
        {
            #[cfg(target_feature = "avx")]
            {
                flags |= Self::AVX;
            }
            #[cfg(target_feature = "avx2")]
            {
                flags |= Self::AVX2;
            }
            #[cfg(target_feature = "f16c")]
            {
                flags |= Self::F16C;
            }
            #[cfg(target_feature = "bmi1")]
            {
                flags |= Self::BMI1;
            }
            #[cfg(target_feature = "bmi2")]
            {
                flags |= Self::BMI2;
            }
            #[cfg(target_feature = "popcnt")]
            {
                flags |= Self::POPCNT;
            }
            #[cfg(target_feature = "sse4.2")]
            {
                flags |= Self::SSE42;
            }
            #[cfg(target_feature = "avx512f")]
            {
                flags |= Self::AVX512F;
            }
        }

        Self { flags }
    }

    #[inline(always)]
    pub const fn has_avx(&self) -> bool {
        (self.flags & Self::AVX) != 0
    }

    #[inline(always)]
    pub const fn has_avx2(&self) -> bool {
        (self.flags & Self::AVX2) != 0
    }

    #[inline(always)]
    pub const fn has_f16c(&self) -> bool {
        (self.flags & Self::F16C) != 0
    }

    #[inline(always)]
    pub const fn has_bmi1(&self) -> bool {
        (self.flags & Self::BMI1) != 0
    }

    #[inline(always)]
    pub const fn has_bmi2(&self) -> bool {
        (self.flags & Self::BMI2) != 0
    }

    #[inline(always)]
    pub const fn has_popcnt(&self) -> bool {
        (self.flags & Self::POPCNT) != 0
    }

    #[inline(always)]
    pub const fn has_sse42(&self) -> bool {
        (self.flags & Self::SSE42) != 0
    }

    #[inline(always)]
    pub const fn has_avx512f(&self) -> bool {
        (self.flags & Self::AVX512F) != 0
    }
}
