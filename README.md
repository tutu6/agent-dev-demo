# AI 私厨助手（POC）

一个用于技术面试展示的 0-1 项目：基于 **FastAPI + LangGraph + LangChain + Qwen + Tavily** 构建多模态「AI 私厨助手」。

## 1. 项目介绍

核心闭环：

1. 用户上传图片（或图片 URL）
2. 多模态模型识别食材（含新鲜度、余量）
3. Tavily 搜索候选菜谱
4. LLM 对候选菜谱打分并输出 Top3（结构化表格）
5. 用户追问（多轮对话，基于 LangGraph checkpoint）
6. 用户提交历史饮食文本，生成周计划

## 2. 架构说明

### 技术栈
- Python 3.11+
- FastAPI（API）
- LangChain（Qwen 调用封装）
- LangGraph（状态机 + checkpoint）
- SQLite（LangGraph SqliteSaver）
- Tavily（检索）
- uv（环境管理）

### 状态设计（State 与 Messages 分离）
`ChefState` 包含：
- Structured State：`ingredients`, `recipes`, `selected_index`, `step`, `question`, `history_text` 等
- Messages：`messages`（用于对话上下文）

### LangGraph 流程
- `intent=analyze`：recognize -> search -> rank
- `intent=followup`：followup
- `intent=weekly_plan`：weekly

使用 `SqliteSaver` + `thread_id` 支持会话恢复。

## 3. 快速启动（uv）

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
cp .env.example .env   # 填入 DASHSCOPE_API_KEY / TAVILY_API_KEY
uv run uvicorn app.main:app --reload --port 8000
```

## 4. 环境变量

创建 `.env`：

```env
DASHSCOPE_API_KEY=your_dashscope_key
TAVILY_API_KEY=your_tavily_key
```

## 5. API 示例

### POST /upload
```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "thread_id=demo-1" \
  -F "image_file=@/path/to/fridge.jpg"
```

### POST /url
```bash
curl -X POST http://127.0.0.1:8000/url \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"demo-1","image_url":"https://.../fridge.jpg"}'
```

### POST /followup
```bash
curl -X POST http://127.0.0.1:8000/followup \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"demo-1","question":"第一个菜怎么做"}'
```

### POST /weekly_plan
```bash
curl -X POST http://127.0.0.1:8000/weekly_plan \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"demo-1","history_text":"过去一周经常吃外卖和甜饮"}'
```

返回中包含：
- `weekly_plan`：结构化周计划（便于第三方平台接入）
- `weekly_plan_markdown`：可直接展示的 markdown 表格

### GET /history/{thread_id}
```bash
curl http://127.0.0.1:8000/history/demo-1
```

## 6. 测试

```bash
uv run pytest -q
```

测试覆盖：
- 图像识别 -> 菜谱检索 -> 排序闭环
- followup 节点
- weekly_plan 节点
- `/upload` `/url` `/followup` `/weekly_plan` `/history/{thread_id}` API 完整性
- base64 异常处理

## 7. 技术亮点

1. **LangGraph 状态机化编排**：意图路由 + 节点职责清晰。
2. **State / Messages 分离**：结构化数据与对话上下文解耦，便于面试讲解。
3. **Sqlite Checkpoint**：多会话恢复能力，低成本可演示。
4. **LLM First 原则**：识别、排序、问答、周计划均由模型驱动，不手写复杂推理算法。
5. **可测试架构**：服务层可替换，单测通过 mock 覆盖核心流程。

## 8. 目录结构

```text
.
├── app/
│   ├── api/          # FastAPI 路由
│   ├── agents/       # Agent 封装
│   ├── core/         # 配置和日志
│   ├── graph/        # LangGraph 工作流和状态
│   ├── schemas/      # Pydantic 请求/响应模型
│   ├── services/     # LLM 和 Tavily 服务
│   └── main.py       # 应用入口
├── tests/            # 单元测试和集成测试
├── resources/        # 资源文件（流程图等）
├── frontend/         # React 前端
├── .env.example      # 环境变量模板
├── pyproject.toml    # Python 项目配置
└── README.md
```
