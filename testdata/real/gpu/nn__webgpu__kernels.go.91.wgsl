
@group(0) @binding(0) var<storage, read> prm: array<u32>;
fn gid(g: vec3<u32>) -> u32 { return g.x + g.y * 4194240u; }
// -inf, computed at run time: a constant expression that evaluates to an
// infinity is a shader-creation error in WGSL.
fn ninf() -> f32 { return bitcast<f32>(0xff800000u | (prm[0] & 0u)); }

@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
fn wm(f: u32, d: u32) -> u32 { return prm[7u + f * 3u + d]; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid(g);
  if (i >= prm[0]) { return; }
  let C = prm[2]; let M = prm[3]; let Cg = prm[4]; let Mg = prm[5];
  let o2 = i % wm(6u, 2u); let o1 = (i / wm(6u, 2u)) % wm(6u, 1u); let o0 = (i / (wm(6u, 2u) * wm(6u, 1u))) % wm(6u, 0u);
  let outPlane = wm(6u, 0u) * wm(6u, 1u) * wm(6u, 2u);
  let m = (i / outPlane) % M; let ni = i / (outPlane * M);
  let grp = m / Mg;
  let inPlane = wm(0u, 0u) * wm(0u, 1u) * wm(0u, 2u);
  let kPlane = wm(1u, 0u) * wm(1u, 1u) * wm(1u, 2u);
  var acc = 0.0;
  if (prm[6] != 0u) { acc = bias[m]; }
  for (var c = 0u; c < Cg; c++) {
    let xb = (ni * C + grp * Cg + c) * inPlane;
    let wb = (m * Cg + c) * kPlane;
    for (var k0 = 0u; k0 < wm(1u, 0u); k0++) {
      let p0 = i32(o0 * wm(2u, 0u) + k0 * wm(3u, 0u)) - i32(wm(4u, 0u));
      if (p0 < 0 || p0 >= i32(wm(0u, 0u))) { continue; }
      for (var k1 = 0u; k1 < wm(1u, 1u); k1++) {
        let p1 = i32(o1 * wm(2u, 1u) + k1 * wm(3u, 1u)) - i32(wm(4u, 1u));
        if (p1 < 0 || p1 >= i32(wm(0u, 1u))) { continue; }
        for (var k2 = 0u; k2 < wm(1u, 2u); k2++) {
          let p2 = i32(o2 * wm(2u, 2u) + k2 * wm(3u, 2u)) - i32(wm(4u, 2u));
          if (p2 < 0 || p2 >= i32(wm(0u, 2u))) { continue; }
          let xi = xb + (u32(p0) * wm(0u, 1u) + u32(p1)) * wm(0u, 2u) + u32(p2);
          let wi = wb + (k0 * wm(1u, 1u) + k1) * wm(1u, 2u) + k2;
          acc += x[xi] * w[wi];
        }
      }
    }
  }
  y[i] = acc;
}