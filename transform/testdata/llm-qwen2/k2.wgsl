
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

const HD4: u32 = 16u;
const KVD4: u32 = 48u;
const CTXKVD4: u32 = 24576u;
const CTX: u32 = 512u;
const GQ: u32 = 3u;
const G: u32 = 4u;
const WG: u32 = 64u;
const S: u32 = 8u;
const H: u32 = 9u;
const SCALE: f32 = 1.25e-01;
@group(0) @binding(0) var<storage, read> q4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kv4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pv: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> ml: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> state: array<u32>;
var<workgroup> qh: array<vec4<f32>, 16>;
var<workgroup> pw: array<f32, 64>;
var<workgroup> part: array<vec4<f32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c = wid.x;
  let h = wid.y;
  let tk = wid.z;
  let kvh = h / GQ;
  let n = state[0] + tk + 1u;
  let p = c * WG + li;
  let th = tk * H + h;
  if (li < HD4) { qh[li] = q4[th * HD4 + li]; }
  workgroupBarrier();
  var s = -3.0e38;
  if (p < n) {
    var a = 0.0;
    let kb = p * KVD4 + kvh * HD4;
    for (var d = 0u; d < HD4; d = d + 1u) {
      a = a + dot(qh[d], kv4[kb + d]);
    }
    s = a * SCALE;
  }
  let m = rmax(li, s);
  let e = select(0.0, exp(s - m), p < n);
  pw[li] = e;
  let l = rsum(li, e);
  let lane = li % HD4;
  var o = vec4<f32>(0.0);
  var u = li / HD4;
  loop {
    if (u >= WG) { break; }
    let pp = c * WG + u;
    if (pp < n) {
      o = o + pw[u] * kv4[CTXKVD4 + pp * KVD4 + kvh * HD4 + lane];
    }
    u = u + G;
  }
  part[li] = o;
  workgroupBarrier();
  if (li < HD4) {
    var tot = part[li];
    for (var k = 1u; k < G; k = k + 1u) { tot = tot + part[li + k * HD4]; }
    pv[(th * S + c) * HD4 + li] = tot;
  }
  if (li == 0u) { ml[th * S + c] = vec2<f32>(m, l); }
}
