//! Hardware-Accelerated CRC32-C (Castagnoli / SSE 4.2).

use crate::caps::CpuCaps;

/// Calculate CRC32-C using hardware instructions when available (`_mm_crc32_u64`),
/// falling back to software bitwise implementation.
#[inline]
pub fn crc32c(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "sse4.2")]
        {
            return unsafe { crc32c_hw(data) };
        }
        #[cfg(not(target_feature = "sse4.2"))]
        {
            if CpuCaps::detect().has_sse42() {
                return unsafe { crc32c_hw(data) };
            }
        }
    }

    crc32c_scalar(data, 0xFFFF_FFFFu32)
}

/// Calculate CRC32-C with pre-detected capability token.
#[inline]
pub fn crc32c_with_caps(data: &[u8], caps: CpuCaps) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if caps.has_sse42() {
            return unsafe { crc32c_hw(data) };
        }
    }

    crc32c_scalar(data, 0xFFFF_FFFFu32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_hw(data: &[u8]) -> u32 {
    use core::arch::x86_64::*;
    let mut crc = 0xFFFF_FFFFu64;
    let mut offset = 0;
    let len = data.len();

    while offset + 8 <= len {
        // SAFETY: Pointer math and read bounded by offset + 8 <= len.
        let val = unsafe { core::ptr::read_unaligned(data.as_ptr().add(offset) as *const u64) };
        crc = _mm_crc32_u64(crc, val);
        offset += 8;
    }

    while offset < len {
        crc = _mm_crc32_u8(crc as u32, data[offset]) as u64;
        offset += 1;
    }

    (!crc) as u32
}

#[inline]
fn crc32c_scalar(data: &[u8], mut crc: u32) -> u32 {
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0x82F6_3B78 & mask);
        }
    }
    !crc
}
