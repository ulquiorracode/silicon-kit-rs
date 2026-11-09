//! Bit Manipulation (BMI1, BMI2, POPCNT).
//!
//! Provides hardware-accelerated branchless rank/select, bit extraction, and population counting.

use crate::caps::CpuCaps;

/// Count set bits (population count) in a 64-bit word.
///
/// Compiles directly to `popcnt` instruction on x86/ARM.
#[inline(always)]
pub const fn popcnt64(val: u64) -> u32 {
    val.count_ones()
}

/// Branchless Bit Zero High (`bzhi` in BMI2): clear bits above index `n`.
///
/// Returns `val & ((1 << n) - 1)`.
#[inline(always)]
pub fn bzhi64(val: u64, n: u32) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "bmi2")]
        {
            return unsafe { core::arch::x86_64::_bzhi_u64(val, n) };
        }
    }

    if n >= 64 {
        val
    } else {
        val & ((1u64 << n) - 1)
    }
}

/// Calculate rank (number of set bits below index `bit_index`) in 2-3 clock cycles.
#[inline(always)]
pub fn rank64(mask: u64, bit_index: u32) -> usize {
    (bzhi64(mask, bit_index)).count_ones() as usize
}

impl CpuCaps {
    #[inline(always)]
    pub fn rank(&self, mask: u64, bit_index: u32) -> usize {
        rank64(mask, bit_index)
    }
}
