// recorder.cjs - the single JavaScript file of this repository.
//
// Justification: the WGSL of the W3C gpuweb CTS only exists as the *result of
// executing* TypeScript test modules (parameter tables expanded through
// `${...}` interpolation inside `t.expectCompileResult(pass, code)` calls).
// No static extraction can expand those, and re-implementing the CTS test
// framework in Go is out of the question. So we run the real CTS modules under
// Node (its own ts-node setup) with the GPU stubbed: a fake GPUDevice records
// every createShaderModule(code) together with the verdict the test expects
// (whether the call sits inside expectGPUError(..., shouldError)), and nothing
// else. Everything downstream (dedupe, files, manifest) is Go.
//
// Usage: node recorder.cjs <cts-dir> <out.jsonl> [query]
// cwd must be <cts-dir> (the CTS requires being run from its root).
'use strict';

const path = require('path');
const fs = require('fs');

const ctsDir = path.resolve(process.argv[2]);
const outFile = path.resolve(process.argv[3]);
const query = process.argv[4] || 'webgpu:shader,validation,*';
process.chdir(ctsDir);

require(path.join(ctsDir, 'src/common/tools/setup-ts-in-node.js'));
const src = p => path.join(ctsDir, 'src', p);

const { DefaultTestFileLoader } = require(src('common/internal/file_loader.ts'));
const { Logger } = require(src('common/internal/logging/logger.ts'));
const { parseQuery } = require(src('common/internal/query/parseQuery.ts'));
const gpuTest = require(src('webgpu/gpu_test.ts'));

const out = fs.openSync(outFile, 'w');
let cur = null; // {id, features:Set}
let shouldError = null; // verdict context set by expectGPUError / expectValidationError

function emit(rec) {
  fs.writeSync(out, JSON.stringify(rec) + '\n');
}

const fakeLimits = new Proxy({}, { get: () => 0x7fffffff });
const fakeFeatures = { has: () => true, size: 0, forEach() {}, [Symbol.iterator]: function* () {} };

const fakeDevice = {
  limits: fakeLimits,
  features: fakeFeatures,
  createShaderModule(desc) {
    const rec = {
      id: cur.id,
      code: desc.code,
      // A module created outside any error scope is expected to succeed.
      fail: shouldError === true,
      pipeline: false,
      constants: false,
      features: [...cur.features],
    };
    const mod = {
      __rec: rec,
      getCompilationInfo: async () => ({ messages: [] }),
    };
    cur.pending.push(rec);
    return mod;
  },
  createComputePipeline(desc) {
    markPipeline(desc && desc.compute);
    return {};
  },
  createRenderPipeline(desc) {
    markPipeline(desc && desc.vertex);
    markPipeline(desc && desc.fragment);
    return {};
  },
  pushErrorScope() {},
  popErrorScope: async () => null,
  queue: {},
};

function markPipeline(stage) {
  if (!stage || !stage.module || !stage.module.__rec) return;
  const r = stage.module.__rec;
  r.pipeline = true;
  if (stage.constants && Object.keys(stage.constants).length > 0) r.constants = true;
  if (shouldError === true) r.fail = true;
}

const fakeProvider = { device: fakeDevice, adapter: { features: fakeFeatures, limits: fakeLimits } };

// Stub the GPU acquisition, record feature requests instead of skipping.
const BS = gpuTest.GPUTestSubcaseBatchState.prototype;
BS.acquireProvider = async () => fakeProvider;
BS.acquireMismatchedProvider = () => undefined;
function recordFeatures(descriptor) {
  const fs_ = Array.isArray(descriptor) ? descriptor : (descriptor && descriptor.requiredFeatures) || [];
  for (const f of fs_) if (cur && f) cur.features.add(String(f));
}
for (const proto of [BS, gpuTest.AllFeaturesMaxLimitsGPUTestSubcaseBatchState.prototype]) {
  proto.selectDeviceOrSkipTestCase = recordFeatures;
  proto.selectDeviceForTextureFormatOrSkipTestCase = () => {};
  proto.selectDeviceForQueryTypeOrSkipTestCase = () => {};
  proto.selectMismatchedDeviceOrSkipTestCase = () => {};
  proto.usesMismatchedDevice = () => {};
}
const TB = gpuTest.GPUTestBase.prototype;
TB.skipIfDeviceDoesNotHaveFeature = function (f) {
  if (cur) cur.features.add(String(f));
};
TB.skipIfDeviceDoesNotSupportQueryType = () => {};
TB.expectGPUError = function (filter, fn, should = true) {
  const saved = shouldError;
  shouldError = filter === 'validation' ? should : saved;
  try {
    return fn();
  } finally {
    shouldError = saved;
  }
};
TB.expectValidationError = function (fn, should = true) {
  return TB.expectGPUError.call(this, 'validation', fn, should);
};

(async () => {
  const loader = new DefaultTestFileLoader();
  const testcases = await loader.loadCases(parseQuery(query));
  const log = new Logger();
  let nCases = 0;
  let nErr = 0;
  for (const tc of testcases) {
    const name = tc.query.toString();
    cur = { id: name, features: new Set(), pending: [] };
    shouldError = null;
    const [rec, res] = log.record(name);
    try {
      await tc.run(rec, []);
    } catch (e) {
      nErr++;
      emit({ id: name, error: String(e && e.message ? e.message : e) });
    }
    for (const r of cur.pending) {
      r.features = [...new Set([...r.features, ...cur.features])];
      emit(r);
    }
    nCases++;
  }
  emit({ summary: true, cases: nCases, errors: nErr });
  fs.closeSync(out);
  process.exit(0);
})().catch(e => {
  console.error(e.stack || String(e));
  process.exit(1);
});
