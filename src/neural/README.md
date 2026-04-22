# Neural Network Module

Modular neural network implemented as GPU compute shaders, for real-time training and inference inside a Vulkan render pipeline.

## Architecture

```
NeuralNetwork (orchestrator)
├── Encoding[] (multiple input encodings; their outputs are concatenated and fed to the MLP)
│   ├── HashGridEncoding   — multi-resolution hash grid (trainable)
│   ├── FrequencyEncoding  — NeRF-style sin/cos positional encoding
│   ├── SHEncoding         — spherical harmonics basis
│   ├── OneBlobEncoding    — Gaussian blob activation
│   └── IdentityEncoding   — pass-through
└── MLP (fixed hidden width = 64, LeakyReLU, Adam optimizer)
```

## Usage

### 1. Configuration

Construct `NeuralNetwork` directly from JSON:

```json
{
  "network": {
    "encoding": [
      { "type": "hashgrid", "inputDim": 3, "numLevels": 8, "featuresPerLevel": 2,
        "tableSize": 16384, "coarsestResolution": 8, "finestResolution": 128 },
      { "type": "sh", "inputDim": 3, "degree": 3 },
      { "type": "identity", "inputDim": 3 }
    ],
    "mlp": {
      "outputSize": 3,
      "hiddenSize": 64,
      "hiddenLayers": 2
    },
    "useEMA": true,
    "emaAlpha": 0.99,
    "learningRate": 0.01,
    "loadPath": "models/foo.vknn",
    "savePath": "models/foo.vknn"
  }
}
```

`encoding` is an array; each entry specifies the encoding for one input field. The sum of all encodings' `outputDim` equals the MLP's `inputSize` and must be <= 64.
If the `encoding` field is omitted, the MLP reads the raw input directly, and `mlp.inputSize` must be given explicitly.

Optional top-level fields:
- `learningRate` — Adam learning rate applied to the MLP and every trainable encoding. Individual encodings may override by setting `"learningRate"` inside their own entry (e.g. hash grid typically wants a higher LR than the MLP).
- `loadPath` / `savePath` — on-disk parameter blob (MLP + trainable encodings). `loadPath` is applied after `initWeights` so the file silently overrides the random init; `savePath` is flushed in the `NeuralNetwork` destructor (after `vkDeviceWaitIdle` in `main.cpp`). Relative paths resolve against the directory of the scene config (via `paths.h::configRelPath`); absolute paths are used as-is. File format: `"VKNN"` magic + `uint32 version=1` + MLP segment + per-encoding payload; shape mismatch vs. current config throws.

### 2. C++ interface

```cpp
json netJson = passJson["network"];
auto network = std::make_unique<NeuralNetwork>(device, netJson);
network->initWeights(42);
network->createPipelines();

// Inside recordCommand:
network->ensureBuffers(sampleCount);  // grow intermediate buffers if needed

// Inference: rawInput[sampleCount * totalRawInputDim] -> output[sampleCount * outputSize]
network->recordForward(cmd, inputBuffer, outputBuffer, sampleCount);

// Training: rawInput + groundTruth -> forward / backward / adam (+ EMA if enabled)
network->recordTrain(cmd, inputBuffer, gtBuffer, sampleCount);

// Read the previous frame's loss
float loss = network->readLoss();
```

### 3. Encoding parameters

| Type | Parameters | OutputDim | Trainable |
|------|------------|-----------|-----------|
| `hashgrid` | `numLevels`, `featuresPerLevel`, `tableSize`, `coarsestResolution`, `finestResolution` | numLevels × featuresPerLevel | Yes |
| `frequency` | `numFreqs` (1-12) | inputDim × numFreqs × 2 | No |
| `sh` | `degree` (0-4); inputDim must be 3 | (degree+1)² | No |
| `oneblob` | `numBins` (2-32), optional `sigma` | inputDim × numBins | No |
| `identity` | none | inputDim | No |

### 4. Dispatch flow

**Training**: encoding forward → MLP forward → zero grads → MLP backward → encoding backward (trainable only) → Adam → (EMA if enabled) → loss readback

**Inference**: encoding forward → MLP forward (uses shadow weights when EMA is enabled)

All encodings write into different slices of the same concat buffer (strided addressing); no separate concat kernel is needed.
The MLP backward's `dInput` reuses the same concat buffer as its destination, where the trainable encodings' backward kernels pick it up.

### 5. Adding a new Encoding

Like `RenderPassFactory`, `Encoding` uses a self-registering factory. To add a new encoding:

1. Derive from `Encoding` (`encoding.h`) with constructor signature `(Device&, const json&)`.
2. Place `REGISTER_ENCODING(T);` in the class declaration (see `hash_grid_encoding.h`).
3. Place `REGISTER_ENCODING_CPP(T, "name");` at file scope in the `.cpp` (`name` is the json `type` value).
4. Implement the required interface:
   - `getInputDim()` / `getOutputDim()` / `typeName()` / `hasTrainableParams()`
   - `createPipelines()` — load the compute shader SPV
   - `recordForward(cmd, rawInput, inputOffset, inputStride, encodedOutput, outputOffset, outputStride, sampleCount)`
   - Trainable encodings additionally implement `recordBackward()`, `recordZeroGrads()`, `recordAdam()`, `initParams()`, `resetAdamState()`, and parameter-buffer accessors. To participate in EMA they must also override `recordForwardWithParams()` to honor the passed-in `paramAddr` instead of their own trained-weights address.

No factory code changes are needed; `/WHOLEARCHIVE:core` in `CMakeLists.txt` keeps the static initializers from being stripped by the linker.

## File map

| File | Description |
|------|-------------|
| `neural_network.h/cpp` | Orchestrator: manages Encoding[] + MLP, intermediate buffers, dispatch chain, EMA |
| `encoding.h` | `Encoding` abstract base + `EncodingFactory` + self-registering macros |
| `mlp.h/cpp` | MLP: params/grads buffers, forward/backward/adam pipelines |
| `{hash_grid,frequency,sh,oneblob,identity}_encoding.h/cpp` | The five encoding implementations |

Corresponding shaders live in `src/shaders/neural/`.
