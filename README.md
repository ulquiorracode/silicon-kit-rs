# silicon-kit

Zero-overhead, safe hardware intrinsics, CPU feature capability tokens, and mechanical sympathy primitives in pure Rust.

[![CI](https://github.com/ulquiorracode/silicon-kit-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/ulquiorracode/silicon-kit-rs/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![no_std](https://img.shields.io/badge/no__std-supported-success)](Cargo.toml)

## Design Philosophy

- **Zero Overhead**: In release builds with `-C target-cpu=native` (or compile-time target features), safe wrappers inline directly to single CPU instructions with 0 runtime checks.
- **Dynamic Capability Tokens**: When compiling for generic targets, inspect CPU features once with `CpuCaps::detect()`, and use monomorphized capability tokens to bypass branching in hot inner loops.
- **Strict `#![no_std]`**: Core primitives operate seamlessly without dynamic allocation or the Rust standard library.
- **Scrooge Systems Mindset**: Contiguous memory layouts, non-temporal streaming stores to avoid L3 cache pollution, and exact 64-byte alignment sympathy.

## Modules & Feature Flags

| Feature | Description | Architecture / Instructions |
| :--- | :--- | :--- |
| `f16c` | Half-precision float conversions (`pack_f32`, `unpack_f16`, `pack_f32x4`) | x86_64 `vcvtps2ph` / `vcvtph2ps` |
| `streaming` | Non-temporal streaming stores & cache prefetch | AVX2 `_mm256_stream_si256` / `_mm_sfence` |
| `bits` | Single-instruction bit manipulations & rank queries | BMI2 `bzhi`, POPCNT `popcnt` |
| `snorm` | Packed 32-bit vertex normals & signs | `A2B10G10R10_SNORM` |
| `crc32` | Hardware-accelerated CRC32-C (Castagnoli) | SSE 4.2 `_mm_crc32_u64` |

## Benchmarks (Intel Core i9-11900H @ 2.50GHz)

```text
Hardware_F16C_vs_Scalar/silicon_kit_caps_token_pack_f32:  977.89 ps  (sub-nanosecond!)
Hardware_F16C_vs_Scalar/silicon_kit_pack_f32_scalar:     1.1186 ns
Streaming_NonTemporal_Copy/silicon_kit_stream_copy_1MB:   30.517 µs  (~32.8 GiB/s)
Streaming_NonTemporal_Copy/std_copy_from_slice_1MB:       33.448 µs
Hardware_BMI2_POPCNT/rank64:                              1.5702 ns
A2B10G10R10_SNORM/pack_normal:                            6.9086 ns
Hardware_CRC32C/silicon_kit_crc32c_64KB:                  5.0896 µs  (~12.6 GiB/s)
```

## License

Licensed under the MIT License.
