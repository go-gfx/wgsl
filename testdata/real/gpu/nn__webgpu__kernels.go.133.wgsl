
@group(0) @binding(0) var<storage, read> prm: array<u32>;
fn gid(g: vec3<u32>) -> u32 { return g.x + g.y * 4194240u; }
// -inf, computed at run time: a constant expression that evaluates to an
// infinity is a shader-creation error in WGSL.
fn ninf() -> f32 { return bitcast<f32>(0xff800000u | (prm[0] & 0u)); }

@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
fn wm(f: u32, d: u32) -> u32 { return prm[4u + f * 3u + d]; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid(g);
  if (i >= prm[0]) { return; }
  let isMax = prm[2] != 0u; let incl = prm[3] != 0u;
  let o2 = i % wm(6u, 2u); let o1 = (i / wm(6u, 2u)) % wm(6u, 1u); let o0 = (i / (wm(6u, 2u) * wm(6u, 1u))) % wm(6u, 0u);
  let outPlane = wm(6u, 0u) * wm(6u, 1u) * wm(6u, 2u);
  let plane = i / outPlane;
  let xb = plane * wm(0u, 0u) * wm(0u, 1u) * wm(0u, 2u);
  var best = ninf();
  var sum = 0.0; var cnt = 0.0; var padCnt = 0.0;
  for (var k0 = 0u; k0 < wm(1u, 0u); k0++) {
    let p0 = i32(o0 * wm(2u, 0u) + k0 * wm(3u, 0u)) - i32(wm(4u, 0u));
    let in0 = p0 >= 0 && p0 < i32(wm(0u, 0u));
    let pad0 = p0 < i32(wm(0u, 0u) + wm(5u, 0u));
    for (var k1 = 0u; k1 < wm(1u, 1u); k1++) {
      let p1 = i32(o1 * wm(2u, 1u) + k1 * wm(3u, 1u)) - i32(wm(4u, 1u));
      let in1 = p1 >= 0 && p1 < i32(wm(0u, 1u));
      let pad1 = p1 < i32(wm(0u, 1u) + wm(5u, 1u));
      for (var k2 = 0u; k2 < wm(1u, 2u); k2++) {
        let p2 = i32(o2 * wm(2u, 2u) + k2 * wm(3u, 2u)) - i32(wm(4u, 2u));
        let in2 = p2 >= 0 && p2 < i32(wm(0u, 2u));
        let pad2 = p2 < i32(wm(0u, 2u) + wm(5u, 2u));
        if (pad0 && pad1 && pad2) { padCnt += 1.0; }
        if (!(in0 && in1 && in2)) { continue; }
        let v = x[xb + (u32(p0) * wm(0u, 1u) + u32(p1)) * wm(0u, 2u) + u32(p2)];
        if (v > best || v != v) { best = v; }
        sum += v; cnt += 1.0;
      }
    }
  }
  if (isMax) { y[i] = best; return; }
  var d = cnt;
  if (incl) { d = padCnt; }
  if (d > 0.0) { y[i] = sum / d; } else { y[i] = 0.0; }
}