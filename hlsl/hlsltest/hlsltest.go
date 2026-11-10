// Package hlsltest runs the HLSL emitted by package hlsl on Direct3D 11 WARP,
// the software rasterizer that ships with Windows (windows-latest in CI), through
// go-mswin/d3d11, and holds every result to WGSL's definition computed in Go: the
// integer cases HLSL leaves undefined (a zero divisor, MIN / -1, shift counts of
// 32 and more), the conversions of f32 to integer at and past the range, and the
// float functions. It also compiles every emitted file with D3DCompile
// (cs_5_0, IEEE strictness, warnings as errors) and checks, in the disassembly,
// that no mul+add was fused into a mad. It is a module of its own so that
// go-gfx/wgsl does not depend on go-mswin/d3d11, as spirv/vulkantest is for
// Vulkan. The tests build on Windows only.
package hlsltest
