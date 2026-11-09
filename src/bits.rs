//! Bit Manipulation (BMI1, BMI2, POPCNT, LZCNT, TZCNT).
//!
//! Provides hardware-accelerated branchless rank/select, bit extraction, deposit,
//! and bit counting operations for high-speed spatial partitioning (BSP, BVH, Octrees)
//! and bitstream serialization.

use crate::caps::CpuCaps;

/// Count set bits (population count) in a 64-bit word.
///
/// Compiles directly to a single `popcnt` instruction on modern CPUs.
#[inline(always)]
pub const fn popcnt64(val: u64) -> u32 {
    val.count_ones()
}

/// Count trailing zeros in a 64-bit word.
///
/// Translates to single-cycle `tzcnt` / `bsf` / ARM `rbit+clz`.
#[inline(always)]
pub const fn tzcnt64(val: u64) -> u32 {
    val.trailing_zeros()
}

/// Count leading zeros in a 64-bit word.
///
/// Translates to single-cycle `lzcnt` / `bsr` / ARM `clz`.
#[inline(always)]
pub const fn lzcnt64(val: u64) -> u32 {
    val.leading_zeros()
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

/// Parallel Bits Extract (`pext` in BMI2).
///
/// Gathers bits from `val` at positions specified by `mask` into contiguous low-order bits.
/// Invaluable for space-filling curves (Morton codes / Z-order), octree morton hashing,
/// and chess bitboards.
#[inline(always)]
pub fn pext64(val: u64, mask: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "bmi2")]
        {
            return unsafe { core::arch::x86_64::_pext_u64(val, mask) };
        }
        #[cfg(all(feature = "std", not(target_feature = "bmi2")))]
        {
            if std::is_x86_feature_detected!("bmi2") {
                return unsafe { core::arch::x86_64::_pext_u64(val, mask) };
            }
        }
    }

    pext64_scalar(val, mask)
}

/// Parallel Bits Deposit (`pdep` in BMI2).
///
/// Scatters contiguous low-order bits of `val` into positions specified by `mask`.
/// Invaluable for Morton code encoding (interleaving 3D coordinate bits for BVH spatial sorting).
#[inline(always)]
pub fn pdep64(val: u64, mask: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "bmi2")]
        {
            return unsafe { core::arch::x86_64::_pdep_u64(val, mask) };
        }
        #[cfg(all(feature = "std", not(target_feature = "bmi2")))]
        {
            if std::is_x86_feature_detected!("bmi2") {
                return unsafe { core::arch::x86_64::_pdep_u64(val, mask) };
            }
        }
    }

    pdep64_scalar(val, mask)
}

/// Calculate rank (number of set bits below index `bit_index`) in 2-3 clock cycles.
#[inline(always)]
pub fn rank64(mask: u64, bit_index: u32) -> usize {
    (bzhi64(mask, bit_index)).count_ones() as usize
}

#[inline]
fn pext64_scalar(val: u64, mut mask: u64) -> u64 {
    let mut res = 0u64;
    let mut out_bit = 0;
    while mask != 0 {
        let bit = mask & (!mask + 1); // isolate lowest set bit
        if (val & bit) != 0 {
            res |= 1 << out_bit;
        }
        out_bit += 1;
        mask &= mask - 1; // clear lowest set bit
    }
    res
}

#[inline]
fn pdep64_scalar(mut val: u64, mut mask: u64) -> u64 {
    let mut res = 0u64;
    while mask != 0 {
        let bit = mask & (!mask + 1);
        if (val & 1) != 0 {
            res |= bit;
        }
        val >>= 1;
        mask &= mask - 1;
    }
    res
}

impl CpuCaps {
    #[inline(always)]
    pub fn rank(&self, mask: u64, bit_index: u32) -> usize {
        rank64(mask, bit_index)
    }

    #[inline(always)]
    pub fn pext(&self, val: u64, mask: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        if self.has_bmi2() {
            return unsafe { core::arch::x86_64::_pext_u64(val, mask) };
        }
        pext64_scalar(val, mask)
    }

    #[inline(always)]
    pub fn pdep(&self, val: u64, mask: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        if self.has_bmi2() {
            return unsafe { core::arch::x86_64::_pdep_u64(val, mask) };
        }
        pdep64_scalar(val, mask)
    }
}
