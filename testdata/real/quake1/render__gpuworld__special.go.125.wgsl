
struct U { rows: array<vec4f, 3>, p: vec4f, q: vec4f, o: vec4f };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var tex: texture_2d<u32>;
@group(0) @binding(2) var cmap: texture_2d<u32>;
@group(0) @binding(3) var<storage, read> faces: array<vec4f>;

struct VIn { @location(0) pos: vec3f, @location(1) uv: vec2f, @location(2) tr: vec4u, @location(3) fi: u32 };
struct VOut {
  @builtin(position) p: vec4f,
  @location(0) @interpolate(linear) uv: vec2f,
  @location(1) @interpolate(flat) tr: vec4u,
  @location(2) @interpolate(flat) fi: u32,
};

@vertex fn vs(v: VIn) -> VOut {
  let vp = vec3f(dot(u.rows[0].xyz, v.pos) + u.rows[0].w,
                 dot(u.rows[1].xyz, v.pos) + u.rows[1].w,
                 dot(u.rows[2].xyz, v.pos) + u.rows[2].w);
  let d = max(vp.z, u.p.y);
  var o: VOut;
  // Half a pixel to the right (u.p.w is 1 / width, in clip units 1 / width * w).
  o.p = vec4f(vp.x / u.p.x + u.p.w * d, vp.y * u.p.z, 0.5 * d, d);
  o.uv = v.uv; o.tr = v.tr; o.fi = v.fi;
  return o;
}

fn wrap(i: i32, n: i32) -> i32 { return ((i % n) + n) % n; }

fn at(tr: vec4u, x: i32, y: i32) -> u32 { return textureLoad(tex, vec2u(tr.x + u32(x), tr.y + u32(y)), 0).r; }

// render.FillSkyPolygon: the front layer (left half, index 0 is clear)
// scrolls twice as fast as the back layer (right half, at half the scale).
@fragment fn fsSky(v: VOut) -> @location(0) vec4u {
  let W = u.q.y; let H = u.q.z;
  let px = floor(v.p.x); let py = floor(v.p.y);
  let temp = max(W, H);
  let wu = 8192.0 * (px - floor(W * 0.5)) / temp;
  let wv = 8192.0 * (floor(H * 0.5) - py) / temp;
  var e = 4096.0 * u.rows[2].xyz + wu * u.rows[0].xyz + wv * u.rows[1].xyz;
  e.z = e.z * 3.0;
  e = normalize(e);
  let off = u.q.w * 8.0;
  let s = i32((off + 378.0 * e.x) * 65536.0);
  let t = i32((off + 378.0 * e.y) * 65536.0);
  let col = (s >> 16u) & 127;
  let row = (t >> 16u) & 127;
  let shift = i32(off);
  let f = at(v.tr, (col + shift) & 127, (row + shift) & 127);
  if (f != 0u) { return vec4u(f, 0u, 0u, 0u); }
  return vec4u(at(v.tr, 128 + col, row), 0u, 0u, 0u);
}

var<private> sintable = array<i32, 256>(0);

@fragment fn fsTurb(v: VOut) -> @location(0) vec4u {
  let f0 = faces[v.fi * 4u]; let f1 = faces[v.fi * 4u + 1u];
  let f2 = faces[v.fi * 4u + 2u]; let f3 = faces[v.fi * 4u + 3u];
  let W = u.q.y; let H = u.q.z;
  let halfW = W * 0.5; let halfH = H * 0.5;
  let X = floor(v.p.x) + 0.5; let Y = floor(v.p.y) + 0.5;
  let right = u.rows[0].xyz; let up = u.rows[1].xyz; let fwd = u.rows[2].xyz;
  let D = fwd + (X - halfW) * u.p.x / halfW * right + (halfH - Y) * u.p.x / halfW * up;
  let org = u.o.xyz;
  let nd = dot(f0.xyz, D);
  let z = (f0.w - dot(f0.xyz, org)) / nd;
  let cs = dot(f1.xyz, org) + f1.w - f3.x;
  let ct = dot(f2.xyz, org) + f2.w - f3.y;
  var s = i32((cs + z * dot(f1.xyz, D)) * 65536.0);
  var t = i32((ct + z * dot(f2.xyz, D)) * 65536.0);
  s = clamp(s, 0, i32(f3.z) * 65536 - 1);
  t = clamp(t, 0, i32(f3.w) * 65536 - 1);
  let off = i32(u.q.x * 20.0) & 127;
  let sturb = s + sintable[off + ((t >> 16u) & 127)];
  let tturb = t + sintable[off + ((s >> 16u) & 127)];
  let tw = i32(v.tr.z); let th = i32(v.tr.w);
  let col = i32((u32(sturb) >> 16u) % u32(tw));
  let row = i32((u32(tturb) >> 16u) % u32(th));
  return vec4u(at(v.tr, col, row), 0u, 0u, 0u);
}
