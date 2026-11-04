
struct U { rows: array<vec4f, 3>, p: vec4f, q: vec4f, o: vec4f };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var tex: texture_2d<u32>;
@group(0) @binding(2) var lm: texture_2d<u32>;
@group(0) @binding(3) var cmap: texture_2d<u32>;
@group(0) @binding(4) var<storage, read> faceTab: array<vec4f>;
@group(0) @binding(5) var<storage, read> faceVerts: array<vec4f>;

struct VIn {
  @location(0) pos: vec3f,
  @location(3) tr: vec4u, @location(4) lr: vec4u, @location(5) fi: u32,
};
struct VOut {
  @builtin(position) p: vec4f,
  @location(2) @interpolate(flat) tr: vec4u, @location(3) @interpolate(flat) lr: vec4u,
  @location(4) @interpolate(flat) fi: u32,
};

@vertex fn vs(v: VIn) -> VOut {
  let vp = vec3f(dot(u.rows[0].xyz, v.pos) + u.rows[0].w,
                 dot(u.rows[1].xyz, v.pos) + u.rows[1].w,
                 dot(u.rows[2].xyz, v.pos) + u.rows[2].w);
  let d = max(vp.z, u.p.y);
  var o: VOut;
  // The reference takes pixel x when its left edge is inside the span
  // (render.FillPerspectiveLightmappedPolygon: ceil(xLeft)): the polygon is
  // moved half a pixel right (u.p.w is 1 / width) so that the GPU takes the
  // same pixels; the fragment then works from its own pixel index.
  o.p = vec4f(vp.x / u.p.x + u.p.w * d, vp.y * u.p.z, 0.5 * d, d);
  o.tr = v.tr; o.lr = v.lr; o.fi = v.fi;
  return o;
}

fn wrap(i: i32, n: i32) -> i32 { return ((i % n) + n) % n; }

// The screen position of a world vertex as bsprender.TransformFaceLightmapped
// gives it: depth clamped to the near plane.
fn proj(w: vec3f) -> vec2f {
  let vp = vec3f(dot(u.rows[0].xyz, w) + u.rows[0].w,
                 dot(u.rows[1].xyz, w) + u.rows[1].w,
                 dot(u.rows[2].xyz, w) + u.rows[2].w);
  let d = max(vp.z, u.p.y);
  let halfW = u.q.y * 0.5;
  let scale = halfW / u.p.x;
  return vec2f(halfW + vp.x * scale / d, u.q.z * 0.5 - vp.y * scale / d);
}

// A polygon vertex as the reference fills it: its screen position, and
// (1/z, u/z, v/z) with z the depth clamped to the near plane.
struct PV { s: vec2f, w: vec3f };

fn polyVertex(p: vec3f, f1: vec4f, f2: vec4f) -> PV {
  let vp = vec3f(dot(u.rows[0].xyz, p) + u.rows[0].w,
                 dot(u.rows[1].xyz, p) + u.rows[1].w,
                 dot(u.rows[2].xyz, p) + u.rows[2].w);
  let d = max(vp.z, u.p.y);
  let halfW = u.q.y * 0.5;
  let scale = halfW / u.p.x;
  let inv = 1.0 / d;
  return PV(vec2f(halfW + vp.x * scale / d, u.q.z * 0.5 - vp.y * scale / d),
            vec3f(inv, (dot(f1.xyz, p) + f1.w) * inv, (dot(f2.xyz, p) + f2.w) * inv));
}

// (u, v, lightmap s, lightmap t) at screen x xc on the run from xl to xr
// whose ends carry wl and wr: linear in (1/z, u/z, v/z), then divided.
fn runAt(xc: f32, xl: f32, wl: vec3f, wr: vec3f, span: f32, minST: vec2f) -> vec4f {
  var dW = vec3f(0.0);
  if (span > 0.0) { dW = (wr - wl) / span; }
  let w = wl + (xc - xl) * dW;
  let z = 1.0 / w.x;
  let uu = w.y * z; let vv = w.z * z;
  return vec4f(uu, vv, (uu - minST.x) / 16.0, (vv - minST.y) / 16.0);
}

@fragment fn fs(v: VOut) -> @location(0) vec4u {
  let f0 = faceTab[v.fi * 4u]; let f1 = faceTab[v.fi * 4u + 1u];
  let f2 = faceTab[v.fi * 4u + 2u]; let f3 = faceTab[v.fi * 4u + 3u];
  let px = i32(floor(v.p.x)); let py = i32(floor(v.p.y));
  let yf = f32(py) + 0.5;
  // The scanline run of the face the reference fills: its first and last
  // pixel, from the polygon's edges at the scanline centre.
  var xl = 1.0e30; var xr = -1.0e30;
  var wl = vec3f(1.0, 0.0, 0.0); var wr = vec3f(1.0, 0.0, 0.0);
  let n = u32(f3.w); let first = u32(f3.z);
  var prev = polyVertex(faceVerts[first + n - 1u].xyz, f1, f2);
  for (var i = 0u; i < n; i++) {
    let cur = polyVertex(faceVerts[first + i].xyz, f1, f2);
    if ((prev.s.y <= yf && cur.s.y > yf) || (cur.s.y <= yf && prev.s.y > yf)) {
      let t = (yf - prev.s.y) / (cur.s.y - prev.s.y);
      let x = prev.s.x + t * (cur.s.x - prev.s.x);
      let w = prev.w + t * (cur.w - prev.w);
      if (x < xl) { xl = x; wl = w; }
      if (x > xr) { xr = x; wr = w; }
    }
    prev = cur;
  }
  var x0 = px; var x1 = px;
  if (xl <= xr) { x0 = max(i32(ceil(xl)), 0); x1 = min(i32(floor(xr)), i32(u.q.y) - 1); }
  // Sub-spans of 8 pixels from the run's first pixel, perspective-exact at
  // their ends and linear between (d_scan.c D_DrawSpans8): the last one ends
  // on the run's last pixel.
  let k = max(px - x0, 0) / 8;
  let xa = x0 + 8 * k;
  let count = x1 - xa + 1;
  var xe = xa + 8; var inv = 0.125;
  if (count <= 8) { xe = xa + count - 1; inv = select(0.0, 1.0 / f32(count - 1), count > 1); }
  let span = xr - xl;
  let A = runAt(f32(xa) + 0.5, xl, wl, wr, span, f3.xy);
  let B = runAt(f32(xe) + 0.5, xl, wl, wr, span, f3.xy);
  let c = A + f32(px - xa) * (B - A) * inv;

  let tw = i32(v.tr.z); let th = i32(v.tr.w);
  let tx = wrap(i32(floor(c.x)), tw);
  let ty = wrap(i32(floor(c.y)), th);
  let texel = textureLoad(tex, vec2u(v.tr.x + u32(tx), v.tr.y + u32(ty)), 0).r;

  let lw = f32(v.lr.z - 1u); let lh = f32(v.lr.w - 1u);
  let ls = clamp(c.z, 0.0, lw); let lt = clamp(c.w, 0.0, lh);
  let s0 = u32(floor(ls)); let t0 = u32(floor(lt));
  let s1 = min(s0 + 1u, v.lr.z - 1u); let t1 = min(t0 + 1u, v.lr.w - 1u);
  let fs_ = ls - f32(s0); let ft = lt - f32(t0);
  let l00 = f32(textureLoad(lm, vec2u(v.lr.x + s0, v.lr.y + t0), 0).r);
  let l10 = f32(textureLoad(lm, vec2u(v.lr.x + s1, v.lr.y + t0), 0).r);
  let l01 = f32(textureLoad(lm, vec2u(v.lr.x + s0, v.lr.y + t1), 0).r);
  let l11 = f32(textureLoad(lm, vec2u(v.lr.x + s1, v.lr.y + t1), 0).r);
  let l0 = l00 + fs_ * (l10 - l00);
  let l1 = l01 + fs_ * (l11 - l01);
  let sample = l0 + ft * (l1 - l0);
  let row = clamp((255 - i32(sample)) >> 2u, 0, 63);
  return vec4u(textureLoad(cmap, vec2u(texel, u32(row)), 0).r, 0u, 0u, 0u);
}
