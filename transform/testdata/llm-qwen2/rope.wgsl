
struct Meta { heads: u32, kvheads: u32, hd: u32, ctx: u32 }
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read> kin: array<f32>;
@group(0) @binding(2) var<storage, read> vin: array<f32>;
@group(0) @binding(3) var<storage, read_write> kc: array<f32>;
@group(0) @binding(4) var<storage, read_write> vc: array<f32>;
@group(0) @binding(5) var<storage, read> state: array<u32>;
@group(0) @binding(6) var<storage, read> cs: array<f32>;
@group(0) @binding(7) var<storage, read> prm: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * (nwg.x * 64u) + gid.x;
  let hp = prm.hd / 2u;
  let nq = prm.heads * hp;
  let nk = prm.kvheads * hp;
  let kvd = prm.kvheads * prm.hd;
  let pos = state[0];
  if (i < nq) {
    let h = i / hp;
    let p = i % hp;
    let c = cs[(pos * hp + p) * 2u];
    let s = cs[(pos * hp + p) * 2u + 1u];
    let a = q[h * prm.hd + 2u * p];
    let b = q[h * prm.hd + 2u * p + 1u];
    q[h * prm.hd + 2u * p] = a * c - b * s;
    q[h * prm.hd + 2u * p + 1u] = a * s + b * c;
  } else if (i < nq + nk) {
    let j = i - nq;
    let h = j / hp;
    let p = j % hp;
    let c = cs[(pos * hp + p) * 2u];
    let s = cs[(pos * hp + p) * 2u + 1u];
    let a = kin[h * prm.hd + 2u * p];
    let b = kin[h * prm.hd + 2u * p + 1u];
    kc[pos * kvd + h * prm.hd + 2u * p] = a * c - b * s;
    kc[pos * kvd + h * prm.hd + 2u * p + 1u] = a * s + b * c;
  } else if (i < nq + nk + kvd) {
    let j = i - nq - nk;
    vc[pos * kvd + j] = vin[j];
  }
}
