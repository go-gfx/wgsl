package spirv

// The GLSL.std.450 extended instruction set: the numbers are the "opcode" of each
// instruction in SPIRV-Headers, include/spirv/unified1/extinst.glsl.std.450.grammar.json
// (FAbs 4, SAbs 5, Floor 8, Sqrt 31, FMin 37, UMin 38, SMin 39, FMax 40, UMax 41,
// SMax 42, FClamp 43, Length 66; only those the emitter uses).
const (
	glslFAbs  = 4
	glslFloor = 8
	glslSqrt  = 31
	glslFMin  = 37
	glslUMin  = 38
	glslSMin  = 39
	glslFMax  = 40
	glslUMax  = 41
	glslSMax  = 42
)
