//! Hardware Fixed-Function Vertex Normal & Tangent Packing.
//!
//! Provides `A2B10G10R10_SNORM` (DXGI_FORMAT_R10G10B10A2_SNORM / VK_FORMAT_A2B10G10R10_SNORM_PACK32).
//! Unpacked by GPU Input Assembler fixed-function silicon into `vec4` registers with **0 fragment shader ALU cycles**
//! and over 10x higher angular fidelity (<0.08° error) than 8-bit formats.

/// 32-bit packed A2B10G10R10 signed normalized normal + tangent sign.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "zerocopy", derive(zerocopy::FromBytes, zerocopy::Immutable, zerocopy::IntoBytes, zerocopy::KnownLayout))]
#[repr(transparent)]
pub struct PackedNormal10(pub u32);

impl PackedNormal10 {
    /// Pack normalized unit normal [x, y, z] and tangent sign (+1.0 or -1.0) into 32-bit A2B10G10R10_SNORM.
    #[inline(always)]
    pub fn pack(normal: [f32; 3], tangent_sign: f32) -> Self {
        let clamp10 = |v: f32| -> u32 {
            let clamped = v.clamp(-1.0, 1.0);
            let scaled_f = clamped * 511.0;
            let rounded = if scaled_f >= 0.0 {
                (scaled_f + 0.5) as i32
            } else {
                (scaled_f - 0.5) as i32
            };
            (rounded & 0x3FF) as u32
        };
        let x = clamp10(normal[0]);
        let y = clamp10(normal[1]);
        let z = clamp10(normal[2]);
        let w = if tangent_sign >= 0.0 { 1u32 } else { 3u32 };
        Self(x | (y << 10) | (z << 20) | (w << 30))
    }

    /// Fast CPU unpack to [f32; 3] and tangent sign (for physics / raytracing BVH builds).
    #[inline(always)]
    pub fn unpack(&self) -> ([f32; 3], f32) {
        let sign_extend_10 = |bits: u32| -> f32 {
            let signed = ((bits as i32) << 22) >> 22;
            (signed as f32) / 511.0
        };
        let nx = sign_extend_10(self.0 & 0x3FF);
        let ny = sign_extend_10((self.0 >> 10) & 0x3FF);
        let nz = sign_extend_10((self.0 >> 20) & 0x3FF);
        let sign = if (self.0 >> 30) == 1 { 1.0 } else { -1.0 };
        ([nx, ny, nz], sign)
    }
}
