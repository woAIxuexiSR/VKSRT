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

## 3. 创建任务

任务是会话级别的，每次新会话需要重新创建：

```
TaskCreate(subject: "完善简单的 ray tracing pass", description: "ray tracing pass 屏幕全黑，需要 debug 修复")
TaskCreate(subject: "实现简单的 tone mapping pass", description: "实现后处理 tone mapping pass，并实现 renderpass 链式串联")
TaskCreate(subject: "实现基于 assimp 的模型导入", description: "用 assimp 导入模型替代 hardcode 的 cornell box")
TaskCreate(subject: "实现更多的材质系统", description: "支持更多材质类型")

TaskUpdate(taskId: "2", addBlockedBy: ["1"])
TaskUpdate(taskId: "3", addBlockedBy: ["2"])
TaskUpdate(taskId: "4", addBlockedBy: ["3"])
```

根据实际进度调整任务内容。
