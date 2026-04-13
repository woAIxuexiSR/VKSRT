# VKSRT Team Setup

每次新会话需要重新启动 team。按以下步骤操作：

## 1. 创建 Team
```
TeamCreate(team_name: "vksrt", description: "VKSRT Vulkan 光线追踪渲染器开发")
```

## 2. 启动 Agents

### Developer
```
Agent(name: "developer", model: "ppio/pa/claude-opus-4-6", team_name: "vksrt", description: "代码实现", prompt: "你的任务是根据 lead 的命令进行代码开发和 git 版本管理。先阅读 .claude/agents/developer.md 了解你的职责，然后阅读项目的 CLAUDE.md 了解项目背景和规范。准备就绪后向 lead (team-lead) 报告，等待任务分配。项目路径：D:\\works\\Vscode\\VKSRT，参考项目：D:\\works\\Vscode\\EasyVulkan。你是 team vksrt 的 developer 角色。完成任务后用 TaskUpdate 标记完成，然后检查 TaskList 寻找下一个可用任务。", run_in_background: true)
```

### Reviewer
```
Agent(name: "reviewer", model: "ppio/pa/claude-opus-4-6", team_name: "vksrt", description: "代码审查", prompt: "你的任务是根据 lead 的命令进行代码审查和 debug。先阅读 .claude/agents/reviewer.md 了解你的职责，然后阅读项目的 CLAUDE.md 了解项目背景和规范。准备就绪后向 lead (team-lead) 报告，等待任务分配。项目路径：D:\\works\\Vscode\\VKSRT，参考项目：D:\\works\\Vscode\\EasyVulkan。你是 team vksrt 的 reviewer 角色。完成任务后用 TaskUpdate 标记完成，然后检查 TaskList 寻找下一个可用任务。", run_in_background: true)
```

## 3. 工作流程（阶段同步）

每个任务/阶段按以下流程推进，完成一个阶段后同步对齐再进入下一个：

1. **Lead 与用户讨论方案** → 确认后分配给 Developer
2. **Developer 实现** → 完成后停下，等待审查
3. **Reviewer 审查** → 发现问题反馈给 lead
4. **Developer 修复** → 修复 reviewer 提出的问题，完成后停下
5. **Reviewer 二次确认** → 确认修复通过
6. **用户验证** → lead 通知用户验证改动，用户确认 OK 后才能提交
7. **Developer 提交** → git commit 统一提交该阶段改动
8. **同步检查点** → 所有人对齐状态，确认后再进入下一阶段

**原则：**
- 不要一股脑地连续执行多个任务，每个阶段完成后必须同步
- **Team Lead**：负责方案讨论、阶段推进和用户沟通，不自己执行任务（不读代码、不写代码），将任务委派给对应 agent
- **Developer**：执行代码实现和修复，审查通过且用户确认后负责 git commit。完成后等待审查，审查通过前不接新任务。严格按 plan.md 或 lead 给出的方案实现，不要自行重新规划
- **Reviewer**：基于 `git diff` 审查代码改动（不需要逐文件完整阅读），重点关注正确性、运行时状态转换、UI 交互逻辑

## 4. 任务管理

任务是会话级别的，每次新会话根据当前进度创建。

**已完成**：M1-M4（基础搭建、光栅化、光追、Path Tracing）、M5（NEE+MIS、Bilateral Filter+G-buffer、TAA）、Pipeline Cache

**当前任务**：

```
TaskCreate(subject: "Light Tracing", description: "实现 Light Tracing：从光源发射光线进行积分，反向的 path tracing。作为新的渲染 pass。修复目前 light tracing pass 中存在的 bug")
TaskCreate(subject: "Wavefront Path Tracing", description: "实现 Wavefront Path Tracing：阶段式的 path tracing，集体发射光线、eval 材质等，减少 GPU 的 thread divergence。作为新的渲染 pass。")
```
