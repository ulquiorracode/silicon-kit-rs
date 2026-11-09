//! Cache Hierarchy & Non-Temporal Streaming Stores.
//!
//! Provides streaming writes directly into system memory controller / PCIe write-combining (WC) buffers,
//! completely bypassing L1/L2/L3 CPU caches to eliminate cache thrashing during multi-megabyte transfers.

use crate::caps::CpuCaps;

/// High-throughput non-temporal copy bypassing CPU caches.
///
/// Uses AVX2 streaming stores (`_mm256_stream_si256`) when destination is 32-byte aligned,
/// flushing directly to memory controllers.
#[inline]
pub fn stream_copy(src: &[u8], dst: &mut [u8]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "Source and destination slices must have identical length"
    );

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        {
            if dst.as_ptr() as usize % 32 == 0 {
                return unsafe { stream_copy_avx2(src, dst) };
            }
        }

        #[cfg(all(feature = "std", not(target_feature = "avx2")))]
        {
            if std::is_x86_feature_detected!("avx2") && dst.as_ptr() as usize % 32 == 0 {
                return unsafe { stream_copy_avx2(src, dst) };
            }
        }
    }

    dst.copy_from_slice(src);
}

/// Explicit memory store fence (`sfence`) ensuring all prior streaming stores are globally visible.
#[inline(always)]
pub fn sfence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_sfence();
    }
}

/// Prefetch memory into CPU cache hierarchy.
#[inline(always)]
pub fn prefetch_read_l1<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
}

impl CpuCaps {
    /// Zero-overhead streaming copy using pre-detected capability token.
    #[inline(always)]
    pub fn stream_copy(&self, src: &[u8], dst: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        if self.has_avx2() && dst.as_ptr() as usize % 32 == 0 {
            unsafe {
                stream_copy_avx2(src, dst);
            }
            return;
        }
        dst.copy_from_slice(src);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn stream_copy_avx2(src: &[u8], dst: &mut [u8]) {
    use core::arch::x86_64::*;

    let len = src.len();
    let mut offset = 0;

    while offset + 32 <= len {
        // SAFETY: Pointer math and loads bounded by offset + 32 <= len.
        let chunk = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset) as *const __m256i) };
        // SAFETY: Caller guarantees dst is 32-byte aligned and dst.len() >= src.len().
        unsafe { _mm256_stream_si256(dst.as_mut_ptr().add(offset) as *mut __m256i, chunk) };
        offset += 32;
    }

    if offset < len {
        dst[offset..].copy_from_slice(&src[offset..]);
    }

    // SAFETY: sfence drains store buffers after streaming stores.
    _mm_sfence();
}
