# Neural Network Module

GPU compute shader 实现的模块化神经网络，用于 Vulkan 渲染管线内的实时训练与推理。

## 架构

```
NeuralNetwork (编排器)
├── Encoding[] (多个输入编码，输出 concat 后作为 MLP 输入)
│   ├── HashGridEncoding   — 多分辨率哈希网格 (trainable)
│   ├── FrequencyEncoding  — NeRF 位置编码 sin/cos
│   ├── SHEncoding         — 球谐基函数
│   ├── OneBlobEncoding    — 高斯 blob 激活
│   └── IdentityEncoding   — 直通
└── MLP (固定 hidden width = 64, LeakyReLU, Adam optimizer)
```

## 使用方式

### 1. 配置

通过 JSON 配置 `NeuralNetwork::Config`：

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
    }
  }
}
```

`encoding` 是数组，每个元素指定一个输入 field 的编码方式。所有 encoding 的 outputDim 之和 = MLP inputSize，必须 <= 64。

也支持旧格式（单个 encoding 对象）。

### 2. C++ 接口

```cpp
// 构造
NeuralNetwork::Config cfg;
// ... 从 JSON 解析填充 cfg.encodings 和 cfg.layers ...
auto network = std::make_unique<NeuralNetwork>(device, cfg);
network->initWeights(42);
network->createPipelines();

// 在 recordCommand 中：
network->ensureBuffers(sampleCount);  // 确保中间 buffer 足够大

// 推理：rawInput[sampleCount * totalRawInputDim] → output[sampleCount * outputSize]
network->recordForward(cmd, inputBuffer, outputBuffer, sampleCount);

// 训练：rawInput + groundTruth → 自动 forward/backward/adam
network->recordTrain(cmd, inputBuffer, gtBuffer, sampleCount);

// 读 loss（上一帧的）
float loss = network->readLoss();
```

### 3. Encoding 类型参数

| Type | 参数 | OutputDim | Trainable |
|------|------|-----------|-----------|
| `hashgrid` | `numLevels`, `featuresPerLevel`, `tableSize`, `coarsestResolution`, `finestResolution` | numLevels × featuresPerLevel | Yes |
| `frequency` | `numFreqs` (1-12) | inputDim × numFreqs × 2 | No |
| `sh` | `degree` (0-4), inputDim 必须为 3 | (degree+1)² | No |
| `oneblob` | `numBins` (2-32), 可选 `sigma` | inputDim × numBins | No |
| `identity` | 无 | inputDim | No |

### 4. Dispatch 流程

**训练**：encoding forward → MLP forward → zero grads → MLP backward → encoding backward (trainable) → Adam → loss readback

**推理**：encoding forward → MLP forward

所有 encoding 写入同一个 concat buffer 的不同 slice（strided 寻址），无额外 concat kernel。

### 5. 自定义 Encoding

实现 `Encoding` 基类（`encoding.h`），需要：
- `getInputDim()` / `getOutputDim()`
- `createPipelines()` — 加载 compute shader SPV
- `recordForward(cmd, rawInput, inputOffset, inputStride, encodedOutput, outputOffset, outputStride, sampleCount)` — strided 读写
- 如果 trainable：`recordBackward()`, `recordZeroGrads()`, `recordAdam()`, `initParams()`, `resetAdamState()`

然后在 `encoding_factory.cpp` 的 `createEncoding()` 中注册。

## 文件说明

| 文件 | 说明 |
|------|------|
| `neural_network.h/cpp` | 编排器：管理 Encoding[] + MLP，中间 buffer，dispatch 链 |
| `encoding.h` | Encoding 抽象基类 |
| `encoding_factory.h/cpp` | `createEncoding()` 工厂函数 |
| `mlp.h/cpp` | MLP：params/grads buffer, forward/backward/adam pipeline |
| `hash_grid_encoding.h/cpp` | 多分辨率哈希网格编码 |
| `frequency_encoding.h/cpp` | NeRF 位置编码 |
| `sh_encoding.h/cpp` | 球谐编码 |
| `oneblob_encoding.h/cpp` | 高斯 blob 编码 |
| `identity_encoding.h/cpp` | 直通编码 |

对应 shader 在 `src/shaders/neural/`。
