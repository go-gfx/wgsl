
@group(0) @binding(0) var<storage, read> prm: array<u32>;
fn gid(g: vec3<u32>) -> u32 { return g.x + g.y * 4194240u; }
// -inf, computed at run time: a constant expression that evaluates to an
// infinity is a shader-creation error in WGSL.
fn ninf() -> f32 { return bitcast<f32>(0xff800000u | (prm[0] & 0u)); }

@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = gid(g);
  if (r >= prm[0]) { return; }
  let L = prm[1]; let inner = prm[2];
  let base = (r / inner) * L * inner + r % inner;
  var mx = ninf();
  for (var j = 0u; j < L; j++) { mx = max(mx, x[base + j * inner]); }
  var s = 0.0;
  for (var j = 0u; j < L; j++) { let e = exp(x[base + j * inner] - mx); y[base + j * inner] = e; s += e; }
  for (var j = 0u; j < L; j++) { y[base + j * inner] = y[base + j * inner] / s; }
}