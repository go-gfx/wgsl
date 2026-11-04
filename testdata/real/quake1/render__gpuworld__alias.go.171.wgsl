var<private> anorms = array<vec3f, 162>();

struct U { rows: array<vec4f, 3>, p: vec4f };
struct Ent { r0: vec4f, r1: vec4f, r2: vec4f, scale: vec4f, so: vec4f, info: vec4u, skin: vec4u, lv: vec4f };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var skins: texture_2d<u32>;
@group(0) @binding(2) var cmap: texture_2d<u32>;
@group(0) @binding(3) var<storage, read> poses: array<u32>;
@group(0) @binding(4) var<storage, read> corners: array<vec4u>;
@group(0) @binding(5) var<storage, read> ents: array<Ent>;

struct VOut {
  @builtin(position) p: vec4f,
  @location(0) @interpolate(linear) uv: vec2f,
  @location(1) @interpolate(linear) light: f32,
  @location(2) @interpolate(flat) skin: vec4u,
};

// R_AliasTransformFinalVert's light: r_ambientlight, less (int)(r_shadelight *
// lightcos) when the normal faces the light, not below zero.
fn vlight(e: Ent, idx: u32) -> f32 {
  let cosv = dot(anorms[min(idx, 161u)], e.lv.xyz);
  var temp = i32(e.so.w);
  if (cosv < 0.0) { temp = max(temp + i32(e.lv.w * cosv), 0); }
  return f32(temp);
}

fn lerped(e: Ent, vi: u32) -> vec4f {
  let pa = poses[e.info.x + vi];
  let pb = poses[e.info.y + vi];
  let t = e.scale.w;
  let a = vec3f(f32(pa & 255u), f32((pa >> 8u) & 255u), f32((pa >> 16u) & 255u));
  let b = vec3f(f32(pb & 255u), f32((pb >> 8u) & 255u), f32((pb >> 16u) & 255u));
  let m = floor(a * (1.0 - t) + b * t + vec3f(0.5));
  var normal = pa >> 24u;
  if (t >= 0.5) { normal = pb >> 24u; }
  return vec4f(m, f32(normal));
}

fn view(e: Ent, m: vec3f) -> vec3f {
  let obj = m * e.scale.xyz + e.so.xyz;
  let w = vec3f(dot(e.r0.xyz, obj) + e.r0.w, dot(e.r1.xyz, obj) + e.r1.w, dot(e.r2.xyz, obj) + e.r2.w);
  return vec3f(dot(u.rows[0].xyz, w) + u.rows[0].w, dot(u.rows[1].xyz, w) + u.rows[1].w, dot(u.rows[2].xyz, w) + u.rows[2].w);
}

fn ndc(vp: vec3f) -> vec2f { return vec2f(vp.x / (u.p.x * vp.z), -vp.y * u.p.z / vp.z); }

@vertex fn vs(@builtin(vertex_index) vidx: u32, @builtin(instance_index) inst: u32) -> VOut {
  let e = ents[inst];
  let base = (vidx / 3u) * 3u;
  let c0 = corners[base]; let c1 = corners[base + 1u]; let c2 = corners[base + 2u];
  let l0 = lerped(e, c0.x); let l1 = lerped(e, c1.x); let l2 = lerped(e, c2.x);
  let v0 = view(e, l0.xyz); let v1 = view(e, l1.xyz); let v2 = view(e, l2.xyz);
  var o: VOut;
  o.p = vec4f(0.0, 0.0, 0.0, 1.0);
  o.uv = vec2f(0.0); o.light = 0.0; o.skin = e.skin;
  if (v0.z < 8.0 || v1.z < 8.0 || v2.z < 8.0) { return o; }
  let s0 = ndc(v0); let s1 = ndc(v1); let s2 = ndc(v2);
  // Screen y is flipped from the clip space y; the winding test is the
  // software one, on screen coordinates (x right, y down).
  let area = (s1.x - s0.x) * (s2.y - s0.y) - (s2.x - s0.x) * (s1.y - s0.y);
  if (area > 0.0) { return o; }
  let k = vidx - base;
  var vp = v0; var c = c0; var nrm = l0.w;
  if (k == 1u) { vp = v1; c = c1; nrm = l1.w; }
  if (k == 2u) { vp = v2; c = c2; nrm = l2.w; }
  o.light = vlight(e, u32(nrm));
  // render.FillTexturedPolygon takes pixel x when its left edge, not its
  // centre, is inside the span, and samples the texture at the centre; so
  // the polygon is moved half a pixel right (u.p.w is 1 / width: half a
  // pixel is 1 / width in clip units) and the fragment samples half a pixel
  // further along.
  o.p = vec4f(vp.x / u.p.x + u.p.w * vp.z, vp.y * u.p.z, 0.5 * vp.z, vp.z);
  o.uv = vec2f(bitcast<f32>(c.y), bitcast<f32>(c.z));
  return o;
}

fn wrap(i: i32, n: i32) -> i32 { return ((i % n) + n) % n; }

@fragment fn fs(v: VOut) -> @location(0) vec4u {
  let uv = v.uv + 0.5 * dpdxFine(v.uv);
  let tx = wrap(i32(floor(uv.x)), i32(v.skin.z));
  let ty = wrap(i32(floor(uv.y)), i32(v.skin.w));
  let texel = textureLoad(skins, vec2u(v.skin.x + u32(tx), v.skin.y + u32(ty)), 0).r;
  // d_polyse.c: acolormap[texel + (light & 0xFF00)], the row clamped here.
  let light = v.light + 0.5 * dpdxFine(v.light);
  let row = u32(clamp((i32(light) & 0xFF00) >> 8u, 0, 63));
  return vec4u(textureLoad(cmap, vec2u(texel, row), 0).r, 0u, 0u, 0u);
}
