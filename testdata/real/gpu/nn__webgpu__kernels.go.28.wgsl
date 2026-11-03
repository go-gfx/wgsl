
@group(0) @binding(0) var<storage, read> prm: array<u32>;
fn gid(g: vec3<u32>) -> u32 { return g.x + g.y * 4194240u; }
// -inf, computed at run time: a constant expression that evaluates to an
// infinity is a shader-creation error in WGSL.
fn ninf() -> f32 { return bitcast<f32>(0xff800000u | (prm[0] & 0u)); }

@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid(g);
  if (i >= prm[0]) { return; }
  var rem = i; var ia = 0u; var ib = 0u;
  for (var d = i32(prm[1]) - 1; d >= 0; d--) {
    let dim = prm[2u + u32(d)];
    let c = rem % dim; rem = rem / dim;
    ia += c * prm[10u + u32(d)]; ib += c * prm[18u + u32(d)];
  }
  let p = a[ia]; let q = b[ib];
  y[i] = 0;
}