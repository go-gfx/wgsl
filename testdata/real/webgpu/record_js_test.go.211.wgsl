
struct U { slot: u32, value: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(1) fn main() { o[u.slot] = u.value; }