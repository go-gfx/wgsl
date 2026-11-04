
var<workgroup> red: array<f32, 64>;

fn rsum(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  var s = 64u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) { red[li] = red[li] + red[li + s]; }
    workgroupBarrier();
    s = s >> 1u;
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

fn rmax(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  var s = 64u / 2u;
  loop {
    if (s == 0u) { break; }
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
    s = s >> 1u;
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

const WG: u32 = 64u;
const TPR: u32 = 8u;
const RPW: u32 = 8u;
const N: u32 = 576u;
const NB: u32 = 18u;
const ROWS: u32 = 3072u;
const NX: u32 = 384u;
const TILE4: u32 = 144u;
const NTILES: u32 = 1u;
const TB: u32 = 18u;
const EPS: f32 = 1e-05;
@group(0) @binding(0) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
const NW: u32 = 55296u;
@group(0) @binding(2) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
const F: u32 = 1536u;

var<workgroup> xs: array<vec4<f32>, 144>;

fn nib(w: u32) -> vec4<f32> { return vec4<f32>(f32(w & 15u), f32((w >> 8u) & 15u), f32((w >> 16u) & 15u), f32((w >> 24u) & 15u)); }
fn nibh(w: u32) -> vec4<f32> { return nib(w >> 4u); }


@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let wg = wid.y * NX + wid.x;
  let tk = wid.z;
  let lr = li / TPR;
  let ln = li % TPR;
  
  let hh = RPW / 2u;
  let orow = wg * hh + lr % hh;
  let valid = orow < F;
  let row = select(orow, F + orow, lr >= hh);
  let r = select(0u, row, valid);
  
  var ss = 0.0;
  var si = li;
  loop {
    if (si >= N / 4u) { break; }
    let v = x4[tk * (N / 4u) + si];
    ss = ss + dot(v, v);
    si = si + WG;
  }
  let rs = 1.0 / sqrt(rsum(li, ss) / f32(N) + EPS);

  var acc = 0.0;
  for (var t = 0u; t < NTILES; t = t + 1u) {
    let base4 = t * TILE4;
    var i = li;
    loop {
      if (i >= TILE4) { break; }
      let gi = base4 + i;
      var v = vec4<f32>(0.0);
      if (gi < N / 4u) { v = x4[tk * (N / 4u) + gi] * rs * vec4<f32>(aux[NW + 4u * gi], aux[NW + 4u * gi + 1u], aux[NW + 4u * gi + 2u], aux[NW + 4u * gi + 3u]); }
      xs[i] = v;
      i = i + WG;
    }
    workgroupBarrier();
    var b = ln;
    loop {
      if (b >= TB) { break; }
      let gb = t * TB + b;
      if (gb < NB) {
        let xb = b * 8u;
        
        let q = qs[r * NB + gb];
        let sc = aux[r * NB + gb];
        var s4 = vec4<f32>(0.0);
        s4 = s4 + (nib(q.x) - 8.0) * xs[xb] + (nibh(q.x) - 8.0) * xs[xb + 4u];
        s4 = s4 + (nib(q.y) - 8.0) * xs[xb + 1u] + (nibh(q.y) - 8.0) * xs[xb + 5u];
        s4 = s4 + (nib(q.z) - 8.0) * xs[xb + 2u] + (nibh(q.z) - 8.0) * xs[xb + 6u];
        s4 = s4 + (nib(q.w) - 8.0) * xs[xb + 3u] + (nibh(q.w) - 8.0) * xs[xb + 7u];
        acc = acc + sc * (s4.x + s4.y + s4.z + s4.w);

      }
      b = b + TPR;
    }
    workgroupBarrier();
  }
  red[li] = acc;
  workgroupBarrier();
  var s = TPR / 2u;
  loop {
    if (s == 0u) { break; }
    if (ln < s) { red[li] = red[li] + red[li + s]; }
    workgroupBarrier();
    s = s >> 1u;
  }
  
  if (li < RPW / 2u) {
    let o = wg * (RPW / 2u) + li;
    if (o < F) {
      let g = red[li * TPR];
      let u = red[(li + RPW / 2u) * TPR];
      act[tk * F + o] = g / (1.0 + exp(-g)) * u;
    }
  }
}
