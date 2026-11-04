
struct Inst { off: vec4f };
@group(0) @binding(0) var<storage, read> insts: array<Inst>;
@group(0) @binding(1) var idxTex: texture_2d<u32>;
@group(0) @binding(2) var palTex: texture_2d<f32>;
@group(0) @binding(3) var palSamp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) inst: u32,
};

@vertex fn vs(@location(0) p: vec2f, @location(1) uv: vec2f, @builtin(instance_index) i: u32) -> VOut {
  var o: VOut;
  let d = insts[i].off;
  o.pos = vec4f(p.x + d.x, p.y, d.y, 1.0);
  o.uv = uv;
  o.inst = i;
  return o;
}

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let t = vec2u(clamp(v.uv * 4.0, vec2f(0.0), vec2f(3.0)));
  let idx = textureLoad(idxTex, t, 0).r;
  let c = textureSample(palTex, palSamp, vec2f((f32(idx) + 0.5) / 256.0, 0.5));
  if (v.inst == 2u) { return vec4f(1.0, 0.0, 1.0, 1.0); }
  return c;
}
