
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

struct Meta { nin: u32, nout: u32, acc: u32, nx: u32 }

@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> dd: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> prm: Meta;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wid.y * prm.nx + wid.x;
  let valid = row < prm.nout;
  let r = select(0u, row, valid);
  let nb = prm.nin / 32u;
  let words = nb * 4u;
  var acc = 0.0;
  var w = li;
  loop {
    if (w >= words) { break; }
    let b = w / 4u;
    let k = w % 4u;
    let word = qs[r * words + w];
    let sc = dd[r * nb + b];
    let xb = b * 32u + 4u * k;
    var s = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
      let bt = (word >> (8u * j)) & 0xffu;
      s = s + f32(i32(bt & 15u) - 8) * x[xb + j] + f32(i32(bt >> 4u) - 8) * x[xb + 16u + j];
    }
    acc = acc + sc * s;
    w = w + 64u;
  }
  let tot = rsum(li, acc);
  if (li == 0u && valid) {
    if (prm.acc == 1u) { y[row] = y[row] + tot; } else { y[row] = tot; }
  }
}
