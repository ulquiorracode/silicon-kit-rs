use criterion::{Criterion, black_box, criterion_group, criterion_main};
use silicon_kit::{bits, caps::CpuCaps, crc32, f16c, snorm, streaming::stream_copy};

fn bench_f16c(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hardware_F16C_vs_Scalar");
    let test_val = core::f32::consts::PI;
    let caps = CpuCaps::detect();

    group.bench_function("silicon_kit_pack_f32_safe_facade", |b| {
        b.iter(|| f16c::pack_f32(black_box(test_val)))
    });

    group.bench_function("silicon_kit_caps_token_pack_f32", |b| {
        b.iter(|| caps.pack_f32(black_box(test_val)))
    });

    group.bench_function("silicon_kit_pack_f32_scalar_fallback", |b| {
        b.iter(|| f16c::pack_f32_scalar(black_box(test_val)))
    });

    group.finish();
}

fn bench_streaming_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Streaming_NonTemporal_Copy");
    let size = 1024 * 1024; // 1 MB
    let src = vec![0xABu8; size];
    let mut dst = vec![0u8; size];

    group.bench_function("silicon_kit_stream_copy_1MB", |b| {
        b.iter(|| stream_copy(black_box(&src), black_box(&mut dst)))
    });

    group.bench_function("std_copy_from_slice_1MB", |b| {
        b.iter(|| dst.copy_from_slice(black_box(&src)))
    });

    group.finish();
}

fn bench_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hardware_BMI2_POPCNT");
    let word = 0xF0F0_AAAA_5555_0123u64;

    group.bench_function("rank64", |b| {
        b.iter(|| bits::rank64(black_box(word), black_box(42)))
    });

    group.finish();
}

fn bench_snorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("A2B10G10R10_SNORM");
    let n = [0.0f32, 1.0f32, 0.0f32];

    group.bench_function("pack_normal", |b| {
        b.iter(|| snorm::PackedNormal10::pack(black_box(n), black_box(1.0)))
    });

    group.finish();
}

fn bench_crc32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hardware_CRC32C");
    let data = vec![0x5Au8; 64 * 1024];

    group.bench_function("silicon_kit_crc32c_64KB", |b| {
        b.iter(|| crc32::crc32c(black_box(&data)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_f16c,
    bench_streaming_copy,
    bench_bits,
    bench_snorm,
    bench_crc32
);
criterion_main!(benches);
