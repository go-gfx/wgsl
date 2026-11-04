
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
const ROWS: u32 = 576u;
const NX: u32 = 72u;
const TILE4: u32 = 144u;
const NTILES: u32 = 1u;
const TB: u32 = 18u;
const EPS: f32 = 1e-05;
@group(0) @binding(0) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> dd: array<f32>;
@group(0) @binding(2) var<storage, read> pv: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> ml: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
const HD4: u32 = 16u;
const S: u32 = 8u;

var<workgroup> xs: array<vec4<f32>, 144>;

fn nib(w: u32) -> vec4<f32> { return vec4<f32>(f32(w & 15u), f32((w >> 8u) & 15u), f32((w >> 16u) & 15u), f32((w >> 24u) & 15u)); }
fn nibh(w: u32) -> vec4<f32> { return nib(w >> 4u); }

fn combine(gi: u32, tk: u32) -> vec4<f32> {
  let h = tk * (N / 4u / HD4) + gi / HD4;
  let d4 = gi % HD4;
  let sn = (state[0] + tk + WG) / WG;
  var mx = -3.0e38;
  for (var c = 0u; c < S; c = c + 1u) {
    if (c >= sn) { break; }
    mx = max(mx, ml[h * S + c].x);
  }
  var l = 0.0;
  var o = vec4<f32>(0.0);
  for (var c = 0u; c < S; c = c + 1u) {
    if (c >= sn) { break; }
    let w = exp(ml[h * S + c].x - mx);
    l = l + w * ml[h * S + c].y;
    o = o + w * pv[(h * S + c) * HD4 + d4];
  }
  return o / l;
}


@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let wg = wid.y * NX + wid.x;
  let tk = wid.z;
  let lr = li / TPR;
  let ln = li % TPR;
  let row = wg * RPW + lr; let valid = row < ROWS;
  let r = select(0u, row, valid);
  
  var acc = 0.0;
  for (var t = 0u; t < NTILES; t = t + 1u) {
    let base4 = t * TILE4;
    var i = li;
    loop {
      if (i >= TILE4) { break; }
      let gi = base4 + i;
      var v = vec4<f32>(0.0);
      if (gi < N / 4u) { v = combine(gi, tk); }
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
        let sc = dd[r * NB + gb];
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
  
  if (ln == 0u && valid) { y[tk * ROWS + row] = y[tk * ROWS + row] + red[li]; }
}
