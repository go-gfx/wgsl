
struct Meta { n: u32 }
@group(0) @binding(0) var<storage, read> tokens: array<u32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> dd: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
@group(0) @binding(5) var<storage, read> prm: Meta;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let tok = tokens[state[1] + wid.z];
  let nb = prm.n / 32u;
  var e = li;
  loop {
    if (e >= prm.n) { break; }
    let b = e / 32u;
    let j = e % 32u;
    let w = qs[(tok * nb + b) * 4u + (j % 16u) / 4u];
    let byte = (w >> (8u * (j % 4u))) & 255u;
    var nv = byte & 15u;
    if (j >= 16u) { nv = byte >> 4u; }
    x[wid.z * prm.n + e] = dd[tok * nb + b] * f32(i32(nv) - 8);
    e = e + 64u;
  }
}
