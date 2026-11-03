
@group(0) @binding(0) var<storage, read> prm: array<u32>;
fn gid(g: vec3<u32>) -> u32 { return g.x + g.y * 4194240u; }
// -inf, computed at run time: a constant expression that evaluates to an
// infinity is a shader-creation error in WGSL.
fn ninf() -> f32 { return bitcast<f32>(0xff800000u | (prm[0] & 0u)); }

@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid(g);
  if (i >= prm[0]) { return; }
  let M = prm[1]; let N = prm[2]; let K = prm[3];
  let n = i % N; let m = (i / N) % M;
  var rem = i / (M * N); var ab = 0u; var bb = 0u;
  for (var d = i32(prm[13]) - 1; d >= 0; d--) {
    let dim = prm[14u + u32(d)];
    let co = rem % dim; rem = rem / dim;
    ab += co * prm[22u + u32(d)]; bb += co * prm[30u + u32(d)];
  }
  var acc = 0.0;
  for (var k = 0u; k < K; k++) {
    acc += a[ab + m * prm[4] + k * prm[5]] * b[bb + k * prm[6] + n * prm[7]];
  }
  var v = bitcast<f32>(prm[11]) * acc;
  if (prm[10] != 0u) { v += bitcast<f32>(prm[12]) * c[m * prm[8] + n * prm[9]]; }
  y[i] = v;
}