
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
const N = 256u;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let row = g.y; let col = g.x;
  if (row >= N || col >= N) { return; }
  var acc = 0.0;
  for (var k = 0u; k < N; k++) { acc += a[row * N + k] * b[k * N + col]; }
  c[row * N + col] = acc;
}