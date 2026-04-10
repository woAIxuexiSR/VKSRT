# VKSRT Team Setup

每次新会话需要重新启动 team。按以下步骤操作：

## 1. 创建 Team
```
TeamCreate(team_name: "vksrt", description: "完成一个基于 vulkan 的光线追踪渲染器")
```

## 2. 启动 Agents

### Developer
```
Agent(name: "developer", model: "ppio/pa/claude-opus-4-6", team_name: "vksrt", description: "代码实现", prompt: "你的任务是根据 lead 的命令进行代码开发。先阅读 .claude/agents/developer.md 了解你的职责，然后阅读项目的 CLAUDE.md 了解项目背景和规范。准备就绪后向 lead (team-lead) 报告，等待任务分配。项目路径：D:\\works\\Vscode\\VKSRT，参考项目：D:\\works\\Vscode\\EasyVulkan。你是 team vksrt 的 developer 角色。完成任务后用 TaskUpdate 标记完成，然后检查 TaskList 寻找下一个可用任务。", run_in_background: true)
```

### Reviewer
```
Agent(name: "reviewer", model: "ppio/pa/claude-opus-4-6", team_name: "vksrt", description: "代码审查", prompt: "你的任务是根据 lead 的命令进行代码审查和 debug。先阅读 .claude/agents/reviewer.md 了解你的职责，然后阅读项目的 CLAUDE.md 了解项目背景和规范。准备就绪后向 lead (team-lead) 报告，等待任务分配。项目路径：D:\\works\\Vscode\\VKSRT，参考项目：D:\\works\\Vscode\\EasyVulkan。你是 team vksrt 的 reviewer 角色。完成任务后用 TaskUpdate 标记完成，然后检查 TaskList 寻找下一个可用任务。", run_in_background: true)
```

### Manager
```
Agent(name: "manager", model: "ppio/pa/claude-opus-4-6", team_name: "vksrt", description: "版本管理", prompt: "你的任务是根据 lead 的命令进行 git 版本管理。先阅读 .claude/agents/manager.md 了解你的职责，然后阅读项目的 CLAUDE.md 了解项目背景和规范。准备就绪后向 lead (team-lead) 报告，等待任务分配。项目路径：D:\\works\\Vscode\\VKSRT。你是 team vksrt 的 manager 角色，负责 git 提交、分支管理和版本控制。完成任务后用 TaskUpdate 标记完成，然后检查 TaskList 寻找下一个可用任务。", run_in_background: true)
```

### Planner
```
Agent(name: "planner", model: "ppio/pa/claude-opus-4-6", team_name: "vksrt", description: "代码规划", prompt: "你的任务是根据 lead 的命令进行代码规划。先阅读 .claude/agents/planner.md 了解你的职责，然后阅读项目的 CLAUDE.md 了解项目背景和规范。准备就绪后向 lead (team-lead) 报告，等待任务分配。项目路径：D:\\works\\Vscode\\VKSRT。你是 team vksrt 的 planner 角色，负责任务规划和协调，任务写到当前文件夹下的 plan.md 文件中。完成任务后用 TaskUpdate 标记完成，然后检查 TaskList 寻找下一个可用任务。", run_in_background: true)
```

## 3. 工作流程（阶段同步）

每个任务/阶段按以下流程推进，完成一个阶段后同步对齐再进入下一个：

1. **Planner 规划** → lead 和用户确认方案
2. **Developer 实现** → 完成后停下，等待审查
3. **Reviewer 审查** → 发现问题反馈给 lead
4. **Developer 修复** → 修复 reviewer 提出的问题，完成后停下
5. **Reviewer 二次确认** → 确认修复通过
6. **用户验证** → lead 通知用户验证改动，用户确认 OK 后才能提交
7. **Manager 提交** → git commit 统一提交该阶段改动
8. **同步检查点** → 所有人对齐状态，确认后再进入下一阶段

**原则：**
- 不要一股脑地连续执行多个任务，每个阶段完成后必须同步
- Team Lead 负责阶段的整体推进和用户沟通，不要自己执行任务，确保每个阶段的输出满足预期
- Developer 完成实现后必须等 reviewer 审查通过并修复完毕，才能接新任务
- Reviewer 的修复意见必须被 developer 实际执行并验证，不能跳过
- Manager 只在阶段完成、审查通过后才做 git commit
- Planner 只负责交往复杂的任务规划和设计，简单任务由 lead 直接分配给 developer， planner 不需要过度关注细节

## 4. 创建任务

任务是会话级别的，每次新会话需要重新创建。
M1-M3 已完成，M3.5（config/UI/模型导入/材质系统）已完成，当前任务：

```
TaskCreate(subject: "将 accumulate pass 解耦", description: "将帧累积逻辑从 ray_tracing_pass 中解耦为独立的 accumulate pass，支持 progressive rendering。累积逻辑独立后可以复用于不同的渲染 pass。")
TaskCreate(subject: "相机 UI 解耦", description: "将相机控制和相机 UI 从 ray_tracing_pass 中解耦出来，作为独立模块。相机信息在 ImGui 中独立显示，不绑定在某个具体 pass 的 UI 中。")
TaskCreate(subject: "实现完整的 path tracing", description: "在 ray_tracing.slang 中实现完整的 path tracing：多次弹射、Russian Roulette、使用 BRDF 接口（eval/sample/pdf）进行材质采样、累积渲染，可以参考 easyvulkan 中的实现。配合解耦后的 accumulate pass 实现 progressive rendering。")

TaskUpdate(taskId: "3", addBlockedBy: ["1", "2"])
```

根据实际进度调整任务内容。
