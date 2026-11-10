//! Cache Hierarchy & Non-Temporal Streaming Stores.
//!
//! Provides streaming writes directly into system memory controller / PCIe write-combining (WC) buffers,
//! completely bypassing L1/L2/L3 CPU caches to eliminate cache thrashing during multi-megabyte transfers.
//! Automatically aligns arbitrary unaligned destination slices (head align -> streaming AVX2/NEON body -> tail).

use crate::caps::CpuCaps;

/// High-throughput non-temporal copy bypassing CPU caches.
///
/// Automatically handles arbitrary slice alignment:
/// 1. Copies unaligned head bytes until `dst` reaches a 32-byte alignment boundary.
/// 2. Executes vectorized non-temporal streaming stores across the bulk 32-byte body.
/// 3. Copies remainder tail bytes.
/// 4. Emits an explicit store fence (`sfence` / `dmb`).
#[inline]
pub fn stream_copy(src: &[u8], dst: &mut [u8]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "Source and destination slices must have identical length"
    );

    // Fast-path for small buffers: CPU cache is faster than stream store setup
    if src.len() < 128 {
        dst.copy_from_slice(src);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        {
            return unsafe { stream_copy_auto_aligned_avx2(src, dst) };
        }

        #[cfg(all(feature = "std", not(target_feature = "avx2")))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return unsafe { stream_copy_auto_aligned_avx2(src, dst) };
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("dmb ishst", options(nostack, preserves_flags));
    }
}

/// Prefetch memory into CPU cache hierarchy.
#[inline(always)]
pub fn prefetch_read_l1<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) ptr,
            options(nostack, preserves_flags, readonly),
        );
    }
}

/// Prefetch memory line for write/prepare (RFO - Request For Ownership).
#[inline(always)]
pub fn prefetch_write_l1<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_ET0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!(
            "prfm pstl1keep, [{0}]",
            in(reg) ptr,
            options(nostack, preserves_flags, readonly),
        );
    }
}

impl CpuCaps {
    /// Zero-overhead streaming copy using pre-detected capability token.
    #[inline(always)]
    pub fn stream_copy(&self, src: &[u8], dst: &mut [u8]) {
        assert_eq!(src.len(), dst.len());
        if src.len() < 128 {
            dst.copy_from_slice(src);
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if self.has_avx2() {
            unsafe {
                stream_copy_auto_aligned_avx2(src, dst);
            }
            return;
        }
        dst.copy_from_slice(src);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn stream_copy_auto_aligned_avx2(src: &[u8], dst: &mut [u8]) {
    use core::arch::x86_64::*;

    let len = src.len();
    let dst_ptr = dst.as_mut_ptr() as usize;

    // 1. Calculate bytes required to reach 32-byte alignment on dst
    let unaligned_prefix = (32 - (dst_ptr & 31)) & 31;
    let prefix = unaligned_prefix.min(len);

    if prefix > 0 {
        dst[..prefix].copy_from_slice(&src[..prefix]);
    }

    let mut offset = prefix;
    let aligned_end = prefix + ((len - prefix) & !31);

    // 2. Stream 32-byte chunks with destination strictly aligned
    while offset < aligned_end {
        // SAFETY: Pointer math and loads bounded by aligned_end <= len.
        let chunk = unsafe { _mm256_loadu_si256(src.as_ptr().add(offset) as *const __m256i) };
        // SAFETY: dst + offset is guaranteed 32-byte aligned by prefix calculation.
        unsafe { _mm256_stream_si256(dst.as_mut_ptr().add(offset) as *mut __m256i, chunk) };
        offset += 32;
    }

    // 3. Process remainder tail
    if offset < len {
        dst[offset..].copy_from_slice(&src[offset..]);
    }

    // SAFETY: sfence drains store buffers after streaming stores.
    _mm_sfence();
}
