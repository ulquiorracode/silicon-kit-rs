use proptest::prelude::*;
use silicon_kit::{
    bits, caps::CpuCaps, crc32, f16c, snorm::PackedNormal10, streaming::stream_copy,
};

#[test]
fn test_f16c_roundtrip_basic() {
    let test_values = [
        0.0f32,
        -0.0f32,
        1.0f32,
        -1.0f32,
        65504.0f32,
        -65504.0f32,
        0.000061035156f32,
    ];
    for &val in &test_values {
        let f16_val = f16c::pack_f32(val);
        let back = f16c::unpack_f16(f16_val);
        assert_eq!(val, back, "Roundtrip failed for {val}");
    }
}

proptest! {
    #[test]
    fn test_f16c_matches_scalar_fallback(val: f32) {
        let hw = f16c::pack_f32(val);
        let sw = f16c::pack_f32_scalar(val);

        if val.is_nan() {
            // Both must be NaN in f16 representation (exponent all 1s, non-zero mantissa)
            assert_eq!(hw & 0x7C00, 0x7C00);
            assert_eq!(sw & 0x7C00, 0x7C00);
            assert_ne!(hw & 0x03FF, 0);
            assert_ne!(sw & 0x03FF, 0);
        } else if val.is_infinite() {
            assert_eq!(hw, sw);
        } else {
            // Allow 1 ULP difference at worst on boundary rounding
            let diff = (hw as i32 - sw as i32).abs();
            prop_assert!(diff <= 1, "HW {hw:#06x} vs SW {sw:#06x} for f32 {val}");
        }
    }

    #[test]
    fn test_bits_rank64_invariant(word: u64, bit: u8) {
        let bit = (bit % 65) as u32;
        let rank = bits::rank64(word, bit);

        let mask = if bit >= 64 { u64::MAX } else { (1u64 << bit) - 1 };
        let expected = (word & mask).count_ones() as usize;
        prop_assert_eq!(rank, expected);
    }

    #[test]
    fn test_snorm_normal_unpack_fidelity(x in -1.0f32..1.0f32, y in -1.0f32..1.0f32, z in -1.0f32..1.0f32) {
        let len = (x * x + y * y + z * z).sqrt();
        if len > 1e-4 {
            let n = [x / len, y / len, z / len];
            let packed = PackedNormal10::pack(n, 1.0);
            let (unpacked_normal, tangent_sign) = packed.unpack();

            let dot = n[0] * unpacked_normal[0] + n[1] * unpacked_normal[1] + n[2] * unpacked_normal[2];
            // Dot product should be very close to 1.0 (> 0.995 => error < 5.7 degrees for 10-bit quantization)
            prop_assert!(dot > 0.995, "Dot product {dot} too low for normal {n:?}");
            prop_assert_eq!(tangent_sign, 1.0);
        }
    }

    #[test]
    fn test_crc32c_hardware_matches_scalar(bytes in proptest::collection::vec(any::<u8>(), 0..2048)) {
        let result = crc32::crc32c(&bytes);
        // Compare with software loop
        let mut expected = 0xFFFF_FFFFu32;
        for &b in &bytes {
            expected ^= b as u32;
            for _ in 0..8 {
                let mask = (expected & 1).wrapping_neg();
                expected = (expected >> 1) ^ (0x82F6_3B78 & mask);
            }
        }
        prop_assert_eq!(result, !expected);
    }
}

#[test]
fn test_streaming_copy_correctness() {
    let sizes = [0, 1, 15, 16, 31, 32, 63, 64, 128, 1024, 65536];
    for &sz in &sizes {
        let src: Vec<u8> = (0..sz).map(|i| (i % 255) as u8).collect();
        let mut dst = vec![0u8; sz + 64]; // ensure capacity & alignment test

        // Find 32-byte aligned subslice in dst
        let offset = (32 - (dst.as_ptr() as usize % 32)) % 32;
        let dst_slice = &mut dst[offset..offset + sz];

        stream_copy(&src, dst_slice);
        assert_eq!(&src[..], dst_slice, "Mismatch at size {sz}");
    }
}

#[test]
fn test_cpu_caps_detection() {
    let caps = CpuCaps::detect();
    println!("Detected CPU features: {caps:?}");
    #[cfg(target_arch = "x86_64")]
    {
        // On modern x86_64 desktop/laptop, sse4.2, popcnt, avx2 are expected
        assert!(caps.has_popcnt());
        assert!(caps.has_sse42());
    }
}
