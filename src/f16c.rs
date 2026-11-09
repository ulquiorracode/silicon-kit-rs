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
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let mantissa = bits & 0x007F_FFFF;
    let exp = ((bits >> 23) & 0xFF) as i32;

    if exp == 255 {
        if mantissa != 0 {
            0x7E00 | ((mantissa >> 13) as u16)
        } else {
            (sign | 0x7C00) as u16
        }
    } else if exp > 142 {
        (sign | 0x7C00) as u16
    } else if exp < 113 {
        if exp < 103 {
            sign as u16
        } else {
            let m = mantissa | 0x0080_0000;
            let shift = 113 - exp;
            let half = m >> (shift + 13);
            (sign | half) as u16
        }
    } else {
        let half_exp = ((exp - 127 + 15) as u32) << 10;
        let half_mantissa = mantissa >> 13;
        (sign | half_exp | half_mantissa) as u16
    }
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
