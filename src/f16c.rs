//! Hardware-Accelerated IEEE 754 Half-Precision Float (F16C) Conversions.

use crate::caps::CpuCaps;

/// Convert a single 32-bit float (`f32`) to a 16-bit half-precision float (`f16` represented as `u16`).
#[inline]
pub fn pack_f32(val: f32) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "f16c")]
        {
            // SAFETY: Verified at compile-time by target-feature
            return unsafe { pack_f32_hw(val) };
        }

        #[cfg(all(feature = "std", not(target_feature = "f16c")))]
        {
            if std::is_x86_feature_detected!("f16c") {
                // SAFETY: Verified by runtime CPUID
                return unsafe { pack_f32_hw(val) };
            }
        }
    }

    pack_f32_scalar(val)
}

/// Convert a 16-bit half float (`u16`) to a 32-bit single-precision float (`f32`).
#[inline]
pub fn unpack_f16(bits: u16) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "f16c")]
        {
            // SAFETY: Verified at compile-time by target-feature
            return unsafe { unpack_f16_hw(bits) };
        }

        #[cfg(all(feature = "std", not(target_feature = "f16c")))]
        {
            if std::is_x86_feature_detected!("f16c") {
                // SAFETY: Verified by runtime CPUID
                return unsafe { unpack_f16_hw(bits) };
            }
        }
    }

    unpack_f16_scalar(bits)
}

/// Vectorized conversion of 4x `f32` floats into 4x `f16` (`[u16; 4]`).
#[inline]
pub fn pack_f32x4(vals: [f32; 4]) -> [u16; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "f16c")]
        {
            return unsafe { pack_f32x4_hw(vals) };
        }

        #[cfg(all(feature = "std", not(target_feature = "f16c")))]
        {
            if std::is_x86_feature_detected!("f16c") {
                return unsafe { pack_f32x4_hw(vals) };
            }
        }
    }

    [
        pack_f32_scalar(vals[0]),
        pack_f32_scalar(vals[1]),
        pack_f32_scalar(vals[2]),
        pack_f32_scalar(vals[3]),
    ]
}

impl CpuCaps {
    /// Zero-overhead F16C pack using pre-detected capability token.
    #[inline(always)]
    pub fn pack_f32(&self, val: f32) -> u16 {
        #[cfg(target_arch = "x86_64")]
        if self.has_f16c() {
            // SAFETY: Token guarantees F16C presence
            return unsafe { pack_f32_hw(val) };
        }
        pack_f32_scalar(val)
    }

    /// Zero-overhead F16C unpack using pre-detected capability token.
    #[inline(always)]
    pub fn unpack_f16(&self, bits: u16) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.has_f16c() {
            // SAFETY: Token guarantees F16C presence
            return unsafe { unpack_f16_hw(bits) };
        }
        unpack_f16_scalar(bits)
    }
}

// ---------------------------------------------------------------------------
// Hardware Intrinsics
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn pack_f32_hw(val: f32) -> u16 {
    use core::arch::x86_64::*;
    let v = _mm_set_ss(val);
    let h = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
    _mm_cvtsi128_si32(h) as u16
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn unpack_f16_hw(bits: u16) -> f32 {
    use core::arch::x86_64::*;
    let h = _mm_cvtsi32_si128(bits as i32);
    let v = _mm_cvtph_ps(h);
    _mm_cvtss_f32(v)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn pack_f32x4_hw(vals: [f32; 4]) -> [u16; 4] {
    use core::arch::x86_64::*;
    unsafe {
        let v = _mm_loadu_ps(vals.as_ptr());
        let h = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
        let mut out = [0u16; 4];
        _mm_storel_epi64(out.as_mut_ptr() as *mut __m128i, h);
        out
    }
}

// ---------------------------------------------------------------------------
// Branchless Scalar Software Fallbacks
// ---------------------------------------------------------------------------

#[inline]
pub fn pack_f32_scalar(val: f32) -> u16 {
    let mut x = val.to_bits();
    let x_sgn = x & 0x8000_0000;
    x ^= x_sgn;

    let o = if x >= 0x4780_0000 {
        if x > 0x7F80_0000 {
            // NaN: preserve quiet NaN payload or set default qNaN bit
            let mant = (x & 0x007F_FFFF) >> 13;
            if mant == 0 {
                0x7E00
            } else {
                0x7C00 | 0x0200 | (mant as u16)
            }
        } else {
            // Infinity or overflow
            0x7C00
        }
    } else if x < 0x3880_0000 {
        // Subnormal or zero in FP16 (< 2^-14).
        // Magic value to align 10 mantissa bits at the bottom of float.
        // As long as IEEE 754 addition is round-to-nearest-even, this aligns and rounds in one step:
        // ((127 - 14) + (23 - 10)) << 23 = 126 << 23 = 0x3F00_0000
        const DENORM_MAGIC: u32 = ((127 - 14) + (23 - 10)) << 23;
        let f = f32::from_bits(x) + f32::from_bits(DENORM_MAGIC);
        f.to_bits().wrapping_sub(DENORM_MAGIC) as u16
    } else {
        let mant_odd = (x >> 13) & 1;
        // update exponent, rounding bias part 1
        x = x
            .wrapping_add(((-112i32) as u32) << 23)
            .wrapping_add(0x0FFF);
        // rounding bias part 2: ties to even
        x = x.wrapping_add(mant_odd);
        (x >> 13) as u16
    };

    ((x_sgn >> 16) as u16) | o
}

#[inline]
pub fn unpack_f16_scalar(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign)
        } else {
            let mut m = mantissa << 13;
            let mut e = 127 - 14;
            while (m & 0x0080_0000) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x007F_FFFF;
            f32::from_bits(sign | ((e as u32) << 23) | m)
        }
    } else if exp == 31 {
        if mantissa == 0 {
            f32::from_bits(sign | 0x7F80_0000)
        } else {
            f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13))
        }
    } else {
        let new_exp = (exp as u32 + 127 - 15) << 23;
        let new_mantissa = mantissa << 13;
        f32::from_bits(sign | new_exp | new_mantissa)
    }
}
