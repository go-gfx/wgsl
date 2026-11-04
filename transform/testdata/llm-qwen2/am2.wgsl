
const NWG: u32 = 64u;
@group(0) @binding(0) var<storage, read> pbv: array<f32>;
@group(0) @binding(1) var<storage, read> pbi: array<u32>;
@group(0) @binding(2) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(3) var<storage, read_write> outv: array<u32>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
var<workgroup> bv: array<f32, 64>;
var<workgroup> bi: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) {
  var best = -3.0e38;
  var bidx = 0xffffffffu;
  var i = li;
  loop {
    if (i >= NWG) { break; }
    let v = pbv[i];
    if (v > best) { best = v; bidx = pbi[i]; }
    i = i + 64u;
  }
  bv[li] = best;
  bi[li] = bidx;
  workgroupBarrier();
  var s = 64u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) {
      let ov = bv[li + s];
      let oi = bi[li + s];
      if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; }
    }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (li == 0u) {
    tokens[state[1] + 1u] = bi[0];
    outv[0] = bi[0];
  }
}
