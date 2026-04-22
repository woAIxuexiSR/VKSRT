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

直接通过 JSON 构造 `NeuralNetwork`：

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
    "emaAlpha": 0.99
  }
}
```

`encoding` 是数组，每个元素指定一个输入 field 的编码方式。所有 encoding 的 outputDim 之和 = MLP inputSize，必须 <= 64。
如果没有 `encoding` 字段，则 MLP 直接读原始输入，`mlp.inputSize` 必须显式给出。

### 2. C++ 接口

```cpp
json netJson = passJson["network"];
auto network = std::make_unique<NeuralNetwork>(device, netJson);
network->initWeights(42);
network->createPipelines();

// 在 recordCommand 中：
network->ensureBuffers(sampleCount);  // 确保中间 buffer 足够大

// 推理：rawInput[sampleCount * totalRawInputDim] → output[sampleCount * outputSize]
network->recordForward(cmd, inputBuffer, outputBuffer, sampleCount);

// 训练：rawInput + groundTruth → 自动 forward/backward/adam (+ EMA 如启用)
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

**训练**：encoding forward → MLP forward → zero grads → MLP backward → encoding backward (trainable) → Adam → (EMA 如启用) → loss readback

**推理**：encoding forward → MLP forward（EMA 启用时使用 shadow 权重）

所有 encoding 写入同一个 concat buffer 的不同 slice（strided 寻址），无额外 concat kernel。
MLP backward 的 dInput 复用同一个 concat buffer 作为落点，供 trainable encoding 的 backward 读取。

### 5. 自定义 Encoding

类似 `RenderPassFactory`，Encoding 使用自注册工厂模式。新增一个 encoding 只需：

1. 继承 `Encoding` 基类（`encoding.h`），构造函数签名 `(Device&, const json&)`；
2. 在类声明中放 `REGISTER_ENCODING(T);`（见 `hash_grid_encoding.h`）；
3. 在 `.cpp` 文件作用域放 `REGISTER_ENCODING_CPP(T, "name");`（`name` 是 json `type` 字段值）；
4. 实现必要接口：
   - `getInputDim()` / `getOutputDim()` / `typeName()` / `hasTrainableParams()`
   - `createPipelines()` — 加载 compute shader SPV
   - `recordForward(cmd, rawInput, inputOffset, inputStride, encodedOutput, outputOffset, outputStride, sampleCount)`
   - trainable 还需：`recordBackward()`, `recordZeroGrads()`, `recordAdam()`, `initParams()`, `resetAdamState()`，以及 param buffer 访问器；如配合 EMA，还需 override `recordForwardWithParams()`，用传入的 `paramAddr` 替代自身存储的训练权重地址。

无需改任何工厂代码，`CMakeLists.txt` 里的 `/WHOLEARCHIVE:core` 保证静态初始化不会被链接器剔除。

## 文件说明

| 文件 | 说明 |
|------|------|
| `neural_network.h/cpp` | 编排器：管理 Encoding[] + MLP，中间 buffer，dispatch 链，EMA |
| `encoding.h` | Encoding 抽象基类 + `EncodingFactory` + 自注册宏 |
| `mlp.h/cpp` | MLP：params/grads buffer, forward/backward/adam pipeline |
| `{hash_grid,frequency,sh,oneblob,identity}_encoding.h/cpp` | 5 种 encoding 实现 |

对应 shader 在 `src/shaders/neural/`。
