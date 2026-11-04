
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

struct Meta { heads: u32, kvheads: u32, hd: u32, ctx: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> outv: array<f32>;
@group(0) @binding(5) var<storage, read> state: array<u32>;
@group(0) @binding(6) var<storage, read> prm: Meta;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wid.x;
  let kvh = h / (prm.heads / prm.kvheads);
  let n = state[0] + 1u;
  let kvd = prm.kvheads * prm.hd;
  let sb = h * prm.ctx;
  var mx = -3.0e38;
  var t = li;
  loop {
    if (t >= n) { break; }
    var s = 0.0;
    for (var d = 0u; d < prm.hd; d = d + 1u) {
      s = s + q[h * prm.hd + d] * kc[t * kvd + kvh * prm.hd + d];
    }
    s = s * prm.scale;
    sc[sb + t] = s;
    mx = max(mx, s);
    t = t + 64u;
  }
  let gmax = rmax(li, mx);
  var sum = 0.0;
  t = li;
  loop {
    if (t >= n) { break; }
    let e = exp(sc[sb + t] - gmax);
    sc[sb + t] = e;
    sum = sum + e;
    t = t + 64u;
  }
  let gsum = rsum(li, sum);
  storageBarrier();
  workgroupBarrier();
  var oc = li;
  loop {
    if (oc >= prm.hd) { break; }
    var o = 0.0;
    for (var u = 0u; u < n; u = u + 1u) {
      o = o + sc[sb + u] * vc[u * kvd + kvh * prm.hd + oc];
    }
    outv[h * prm.hd + oc] = o / gsum;
    oc = oc + 64u;
  }
}
