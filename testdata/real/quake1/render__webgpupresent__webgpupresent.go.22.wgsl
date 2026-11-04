
@group(0) @binding(0) var idx: texture_2d<u32>;
@group(0) @binding(1) var pal: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let d = vec2u(textureDimensions(idx));
  let t = min(vec2u(v.uv * vec2f(d)), d - vec2u(1u));
  let k = textureLoad(idx, t, 0).r;
  return vec4f(textureLoad(pal, vec2u(k, 0u), 0).rgb, 1.0);
}
