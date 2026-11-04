
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

struct Meta { n: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> prm: Meta;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) {
  var ss = 0.0;
  var i = li;
  loop {
    if (i >= prm.n) { break; }
    ss = ss + x[i] * x[i];
    i = i + 64u;
  }
  let tot = rsum(li, ss);
  let s = 1.0 / sqrt(tot / f32(prm.n) + prm.eps);
  i = li;
  loop {
    if (i >= prm.n) { break; }
    y[i] = x[i] * s * w[i];
    i = i + 64u;
  }
}
