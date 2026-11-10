package msl

// kernelSrc is go-gfx/gpu2d's raster kernel (kernel.wgsl at gpu2d v0.7.0): the case
// the compute-entry support exists for.
const kernelSrc = `
// The whole raster kernel as one WGSL compute shader. One invocation per
// pixel: it walks the instances binned into its 16 x 16 tile in draw order,
// computes gpu2d.Coverage at its centre in f32 (the functions of
// webgpu/shader.wgsl, step for step), and blends in integers exactly as
// PixelPainter's blendInto does, floored alpha included. It is the arithmetic
// of metal/kernel.metal, so a frame is byte-identical to List.Rasterize over
// any ground (docs/NATIVE.md, decision 1).
//
// Written inside what the other emitters will express (docs/NATIVE.md): four
// storage buffers and one uniform, a 16 x 8 workgroup, pixels as array<u32>,
// blend returning a vector instead of taking references.

struct Inst {
  x: i32, y: i32, w: i32, h: i32,
  cx0: i32, cy0: i32, cx1: i32, cy1: i32,
  color: u32, radius: i32, kind: u32,
  data: u32, stride: u32, srcW: u32, srcH: u32,
  pad: u32,
}

// Frame: the surface size in pixels, the tiles across, and the tile side.
struct Frame {
  w: u32, h: u32, across: u32, tile: u32,
}

@group(0) @binding(0) var<storage, read> insts: array<Inst>;
// bins is List.Bin's spans (two words per tile: the first entry of the tile in
// the list, and how many) followed by its list (instance indices, each tile's
// in draw order).
@group(0) @binding(1) var<storage, read> bins: array<u32>;
@group(0) @binding(2) var<storage, read> arena: array<u32>; // masks and images
// px is one little-endian word per pixel: r, g, b, a from the lowest byte.
@group(0) @binding(3) var<storage, read_write> px: array<u32>;
@group(0) @binding(4) var<uniform> f: Frame;

// byteAt reads byte i of the arena (little-endian words).
fn byteAt(i: u32) -> u32 {
  return (arena[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

// cornerCenter: xy is the circle centre of the corner box (x, y) lies in,
// z is 1 when it lies in one.
fn cornerCenter(x: i32, y: i32, s: Inst, rad: i32) -> vec3f {
  let left = x < s.x + rad;
  let right = x >= s.x + s.w - rad;
  let top = y < s.y + rad;
  let bottom = y >= s.y + s.h - rad;
  if (left && top) { return vec3f(f32(s.x + rad), f32(s.y + rad), 1.0); }
  if (right && top) { return vec3f(f32(s.x + s.w - rad), f32(s.y + rad), 1.0); }
  if (left && bottom) { return vec3f(f32(s.x + rad), f32(s.y + s.h - rad), 1.0); }
  if (right && bottom) { return vec3f(f32(s.x + s.w - rad), f32(s.y + s.h - rad), 1.0); }
  return vec3f(0.0);
}

fn coverage(x: i32, y: i32, s: Inst) -> f32 {
  let rad = s.radius;
  let c = cornerCenter(x, y, s, rad);
  let d = length(vec2f(f32(x) + 0.5 - c.x, f32(y) + 0.5 - c.y));
  if (s.kind == 0u) {
    if (c.z == 0.0) { return 1.0; }
    return clamp(f32(rad) + 0.5 - d, 0.0, 1.0);
  }
  if (rad > 0 && c.z != 0.0) {
    return max(1.0 - abs(d - (f32(rad) - 0.5)), 0.0);
  }
  let edge = y == s.y || y == s.y + s.h - 1 || x == s.x || x == s.x + s.w - 1;
  return select(0.0, 1.0, edge);
}

// blend composites (sr, sg, sb) at alpha ca over dst (r, g, b, a) exactly as
// PixelPainter's blendInto: opaque overwrites, 0 leaves, else rounded colour
// and floored alpha.
fn blend(dst: vec4<u32>, sr: u32, sg: u32, sb: u32, ca: u32) -> vec4<u32> {
  if (ca == 255u) { return vec4<u32>(sr, sg, sb, 255u); }
  if (ca == 0u) { return dst; }
  let ia = 255u - ca;
  return vec4<u32>(
    (sr * ca + dst.x * ia + 127u) / 255u,
    (sg * ca + dst.y * ia + 127u) / 255u,
    (sb * ca + dst.z * ia + 127u) / 255u,
    ca + dst.w * ia / 255u);
}

@compute @workgroup_size(16, 8)
fn raster(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= f.w || gid.y >= f.h) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let down = (f.h + f.tile - 1u) / f.tile;
  let t = (gid.y / f.tile) * f.across + gid.x / f.tile;
  let first = bins[2u * t];
  let count = bins[2u * t + 1u];
  let list = 2u * f.across * down + first;
  let i = gid.y * f.w + gid.x;
  let p = px[i];
  var d = vec4<u32>(p & 255u, (p >> 8u) & 255u, (p >> 16u) & 255u, p >> 24u);
  for (var k = 0u; k < count; k++) {
    let s = insts[bins[list + k]];
    if (x < max(s.x, s.cx0) || x >= min(s.x + s.w, s.cx1) || y < max(s.y, s.cy0) || y >= min(s.y + s.h, s.cy1)) {
      continue;
    }
    let sr = s.color & 255u;
    let sg = (s.color >> 8u) & 255u;
    let sb = (s.color >> 16u) & 255u;
    var ca = s.color >> 24u;
    if (s.kind == 2u || s.kind == 3u) {
      // A mask's coverage, or a path's stored alpha.
      var m = byteAt(s.data + u32(y - s.y) * s.stride + u32(x - s.x));
      if (m == 0u) { continue; }
      if (s.kind == 2u && ca != 255u) { m = m * ca / 255u; }
      d = blend(d, sr, sg, sb, m);
      continue;
    }
    if (s.kind == 4u) {
      // Nearest neighbour, PixelPainter's integer mapping.
      let sx = u32(x - s.x) * s.srcW / u32(s.w);
      let sy = u32(y - s.y) * s.srcH / u32(s.h);
      let o = s.data + (sy * s.srcW + sx) * 4u;
      d = blend(d, byteAt(o), byteAt(o + 1u), byteAt(o + 2u), byteAt(o + 3u));
      continue;
    }
    let cov = coverage(x, y, s);
    if (cov <= 0.0) { continue; }
    if (cov < 1.0) { ca = u32(f32(ca) * cov + 0.5); }
    d = blend(d, sr, sg, sb, ca);
  }
  px[i] = d.x | (d.y << 8u) | (d.z << 16u) | (d.w << 24u);
}
`
