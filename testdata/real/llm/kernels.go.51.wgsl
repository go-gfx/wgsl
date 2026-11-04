
struct Meta { n: u32 }
@group(0) @binding(0) var<storage, read> tokens: array<u32>;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> dd: array<f32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
@group(0) @binding(5) var<storage, read> prm: Meta;

// Q8_0 row gather: element e of the token's row is d * int8.
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) {
  let tok = tokens[state[1]];
  let nb = prm.n / 32u;
  var e = li;
  loop {
    if (e >= prm.n) { break; }
    let b = e / 32u;
    let w = qs[(tok * nb + b) * 8u + (e % 32u) / 4u];
    let v = extractBits(bitcast<i32>(w), 8u * (e % 4u), 8u);
    x[e] = dd[tok * nb + b] * f32(v);
    e = e + 64u;
  }
}
