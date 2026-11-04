
struct Meta { n: u32 }
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> prm: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < prm.n) {
    let v = g[i];
    g[i] = v / (1.0 + exp(-v)) * u[i];
  }
}
