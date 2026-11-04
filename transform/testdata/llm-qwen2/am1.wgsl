
const VOCAB: u32 = 49152u;
const SL: u32 = 768u;
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> pbv: array<f32>;
@group(0) @binding(2) var<storage, read_write> pbi: array<u32>;
var<workgroup> bv: array<f32, 64>;
var<workgroup> bi: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let lo = wid.x * SL;
  let hi = min(lo + SL, VOCAB);
  var best = -3.0e38;
  var bidx = lo;
  var i = lo + li;
  loop {
    if (i >= hi) { break; }
    let v = logits[i];
    if (v > best) { best = v; bidx = i; }
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
  if (li == 0u) { pbv[wid.x] = bv[0]; pbi[wid.x] = bi[0]; }
}
