@group(0) @binding(0) var<storage, read> meta: array<u32>;
@compute @workgroup_size(1) fn main() { _ = meta[0]; }