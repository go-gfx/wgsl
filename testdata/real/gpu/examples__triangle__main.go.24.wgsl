
struct U { angle: f32 };
@group(0) @binding(0) var<uniform> u: U;

struct Out { @builtin(position) pos: vec4<f32>, @location(0) color: vec3<f32> };

@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
  var p = array<vec2<f32>, 3>(vec2(0.0, 0.7), vec2(-0.6, -0.5), vec2(0.6, -0.5));
  var c = array<vec3<f32>, 3>(vec3(1.0, 0.2, 0.2), vec3(0.2, 1.0, 0.2), vec3(0.2, 0.3, 1.0));
  let s = sin(u.angle); let k = cos(u.angle);
  let v = p[i];
  var o: Out;
  o.pos = vec4(v.x * k - v.y * s, v.x * s + v.y * k, 0.0, 1.0);
  o.color = c[i];
  return o;
}

@fragment fn fs(in: Out) -> @location(0) vec4<f32> { return vec4(in.color, 1.0); }
