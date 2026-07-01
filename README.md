# EduRAG — 传智教育智能问答系统

基于 **RAG（检索增强生成）** 技术的智能教育问答平台，融合 MySQL 结构化知识库与 Milvus 向量知识库，集成 BM25 关键词检索、BERT 意图识别、多策略检索增强与大模型流式生成，为 IT 培训教育场景提供精准、高效、可扩展的问答服务。

## 项目架构

```
integrated_qa_system/
├── app.py                          # FastAPI Web 服务入口（HTTP + WebSocket）
├── new_main.py                     # 集成问答核心系统 V2（含历史对话 + 流式输出）
├── old_main.py                     # 集成问答核心系统 V1（基础版）
├── config.ini                      # 全局配置文件（MySQL / Redis / Milvus / LLM）
├── requirements.txt                # Python 依赖清单
├── static/                         # 前端静态资源
│   ├── index.html                  # Web 聊天界面（支持 Markdown / 暗色模式 / 流式输出）
│   └── old_index.html              # 旧版前端页面
│
├── base/                           # 基础设施层
│   ├── config.py                   # 配置管理（支持环境变量覆盖）
│   └── logger.py                   # 日志管理（控制台 + 文件双输出）
│
├── mysql_qa/                       # MySQL 问答子系统（结构化 FQA）
│   ├── main.py                     # MySQL 问答入口
│   ├── db/mysql_client.py          # MySQL 数据库客户端
│   ├── cache/redis_client.py       # Redis 缓存客户端
│   ├── retrieval/bm25_search.py    # BM25 关键词搜索 + Softmax 归一化
│   └── utils/preprocess.py         # Jieba 中文分词预处理
│
├── rag_qa/                         # RAG 问答子系统（非结构化知识库）
│   ├── main.py                     # RAG 系统入口（支持数据处理 / 交互查询双模式）
│   ├── core/
│   │   ├── vector_store.py         # Milvus 向量存储（BGE-M3 稠密+稀疏混合检索 + BGE-Reranker 重排序）
│   │   ├── rag_system.py           # RAG 核心逻辑 V1（基础版）
│   │   ├── new_rag_system.py       # RAG 核心逻辑 V2（含历史对话）
│   │   ├── query_classifier.py     # BERT 查询分类器（通用知识 / 专业咨询）
│   │   ├── strategy_selector.py    # LLM 检索策略选择器（4 种策略自动选优）
│   │   ├── document_processor.py   # 文档加载 + 分层切分（父块→子块）
│   │   └── prompts.py              # Prompt 模板管理（RAG / HyDE / 子查询 / 回溯）
│   ├── edu_document_loaders/       # 教育场景专用文档加载器
│   │   ├── edu_pdfloader.py        # PDF 加载（含 OCR）
│   │   ├── edu_docloader.py        # Word 文档加载（含 OCR）
│   │   ├── edu_pptloader.py        # PPT 文档加载（含 OCR）
│   │   ├── edu_imgloader.py        # 图片加载（OCR 识别）
│   │   └── edu_ocr.py              # OCR 引擎封装
│   ├── edu_text_spliter/           # 教育场景专用文本分割器
│   │   ├── edu_chinese_recursive_text_splitter.py  # 中文递归文本分割
│   │   └── edu_model_text_spliter.py               # 阿里中文优化分割器
│   ├── rag_assessment/             # RAG 质量评估
│   │   ├── rag_as.py               # RAGAS 评估脚本（4 项指标）
│   │   └── rag_evaluate_data.json  # 评估数据集
│   └── classify_data/              # BERT 分类器训练数据
│       ├── model_generic_5000.json # 5000 条训练样本
│       └── 提示词模块.txt           # 分类提示词参考
```

## 核心流程

```
用户问题
   │
   ▼
┌──────────────────┐
│  FastAPI Web 服务  │  ← HTTP POST /api/query 或 WebSocket /api/stream
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  问候语正则匹配    │  → 命中 → 直接返回预设回复
└────────┬─────────┘
         │ 未命中
         ▼
┌──────────────────┐
│  BM25 + MySQL     │  → 分数 ≥ 0.85 → 返回缓存的 FQA 答案
│  关键词检索        │
└────────┬─────────┘
         │ 分数 < 0.85
         ▼
┌──────────────────┐
│  BERT 查询分类器  │  → 通用知识 → LLM 直接回复
│  (意图识别)       │  → 专业咨询 → 进入 RAG 流程
└────────┬─────────┘
         │ 专业咨询
         ▼
┌──────────────────┐
│  LLM 策略选择器   │  4 种策略自动选优：
│                  │  ① 直接检索  ② HyDE 假设答案检索
│                  │  ③ 子查询拆分 ④ 回溯问题简化
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Milvus 混合检索  │  稠密向量 (BGE-M3) + 稀疏向量 + Reranker
│  + 重排序         │  父块→子块分层检索，按学科过滤
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM 生成答案     │  DashScope (Qwen) 流式输出
│  (流式 Token)     │  结合对话历史（最近 5 轮）生成上下文感知回复
└────────┬─────────┘
         │
         ▼
    返回给前端（WebSocket 流式 / HTTP 同步）
```

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| **Web 框架** | FastAPI + Uvicorn（异步高性能） |
| **通信协议** | HTTP RESTful + WebSocket（流式输出） |
| **前端** | 原生 HTML/CSS/JS（暗色模式 / Markdown 渲染 / 会话管理） |
| **关系型数据库** | MySQL（结构化 FQA 问答对 + 对话历史存储） |
| **缓存** | Redis（BM25 问题缓存 / 答案缓存） |
| **向量数据库** | Milvus（稠密 + 稀疏混合向量存储与检索） |
| **嵌入模型** | BGE-M3（1024 维稠密 + 稀疏向量，支持多语言） |
| **重排序模型** | BGE-Reranker-Large（Cross-Encoder 相关性排序） |
| **分类模型** | BERT-Base-Chinese（查询意图二分类） |
| **大语言模型** | 阿里云 DashScope API（Qwen-Plus，兼容 OpenAI 接口） |
| **本地评估 LLM** | Ollama + Qwen2.5:7B（RAGAS 评估） |
| **文档处理** | LangChain + PyMuPDF + python-docx + python-pptx + RapidOCR |
| **文本分割** | 中文递归分割器 + Markdown 分割器（父块 1200 / 子块 300） |
| **关键词检索** | BM25Okapi + Softmax 归一化 |
| **RAG 评估** | RAGAS（忠实度 / 答案相关性 / 上下文精确率 / 上下文召回率） |

## 快速开始

### 环境要求

- Python 3.10+
- MySQL 5.7+
- Redis 5.0+
- Milvus 2.5+（需提前创建数据库 `itcast`）
- 阿里云 DashScope API Key

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Zhengheng01/EduRAg.git
cd EduRAg

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载模型文件（放置在 rag_qa/models/ 目录下）
#    - BGE-M3: https://huggingface.co/BAAI/bge-m3
#    - BGE-Reranker-Large: https://huggingface.co/BAAI/bge-reranker-large
#    - BERT-Base-Chinese: https://huggingface.co/google-bert/bert-base-chinese
#    - BERT-Query-Classifier: 训练后的分类模型
#
#    目录结构示例：
#    rag_qa/models/
#    ├── bge-m3/                    # BGE-M3 嵌入模型
#    ├── bge-reranker-large/        # BGE 重排序模型
#    ├── bert-base-chinese/         # BERT 基座模型
#    └── bert_query_classifier/     # 查询分类器模型
```

### 配置说明

修改项目根目录下的 `config.ini` 文件，或通过环境变量覆盖：

```ini
[mysql]
host = localhost          # 环境变量: MYSQL_HOST
user = root               # 环境变量: MYSQL_USER
password = 123456         # 环境变量: MYSQL_PASSWORD
database = subjects_kg    # 环境变量: MYSQL_DATABASE

[redis]
host = localhost          # 环境变量: REDIS_HOST
port = 6379               # 环境变量: REDIS_PORT
password = 1234           # 环境变量: REDIS_PASSWORD

[milvus]
host = localhost           # 环境变量: MILVUS_HOST
port = 19530               # 环境变量: MILVUS_PORT
database_name = itcast     # 环境变量: MILVUS_DATABASE_NAME
collection_name = edurag_final

[llm]
model = qwen-plus
dashscope_api_key = your_api_key_here    # 环境变量: DASHSCOPE_API_KEY
dashscope_base_url = https://dashscope.aliyuncs.com/compatible-mode/v1

[retrieval]
parent_chunk_size = 1200
child_chunk_size = 300
chunk_overlap = 50
retrieval_k = 3
candidate_m = 2

[app]
valid_sources = ["ai", "java", "test", "ops", "bigdata"]
customer_service_phone = 12345678
```

### 数据准备

**1. MySQL 结构化数据导入**

确保 MySQL 中存在 `subjects_kg` 数据库，将 CSV 格式的学科知识问答数据导入 `jpkb` 表：

```sql
CREATE TABLE IF NOT EXISTS jpkb (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subject_name VARCHAR(20),
    question VARCHAR(1000),
    answer VARCHAR(1000)
);
```

**2. 向量知识库构建**

将各学科的文档（PDF / DOCX / PPT / MD / 图片）按学科分类放入 `rag_qa/data/` 目录：

```
rag_qa/data/
├── ai_data/          # AI 学科文档
├── java_data/        # Java 学科文档
├── test_data/        # 测试学科文档
├── ops_data/         # 运维学科文档
└── bigdata_data/     # 大数据学科文档
```

运行数据处理模式，将文档向量化后存入 Milvus：

```bash
cd rag_qa
python main.py --data-processing --data-dir ./data
```

### 启动服务

```bash
# Web 服务模式（默认端口 8080，可通过 HOST/PORT 环境变量修改）
python app.py

# 命令行交互模式
python new_main.py
```

访问 `http://localhost:8080` 打开 Web 聊天界面。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 前端聊天界面 |
| `POST` | `/api/create_session` | 创建新会话，返回 session_id |
| `POST` | `/api/query` | 同步问答（返回完整答案或提示转流式） |
| `WebSocket` | `/api/stream` | 流式问答（逐 Token 推送） |
| `GET` | `/api/history/{session_id}` | 获取会话历史（最近 5 轮） |
| `DELETE` | `/api/history/{session_id}` | 清除会话历史 |
| `GET` | `/api/sources` | 获取支持的学科类别列表 |
| `GET` | `/health` | Kubernetes 健康探针 |

### WebSocket 消息格式

**发送：**
```json
{
    "query": "AI课程学什么？",
    "source_filter": "ai",
    "session_id": "uuid-xxx"
}
```

**接收（逐 Token 推送）：**
```json
{"type": "start", "session_id": "uuid-xxx"}
{"type": "token", "token": "AI", "session_id": "uuid-xxx"}
{"type": "token", "token": "课程", "session_id": "uuid-xxx"}
{"type": "end", "session_id": "uuid-xxx", "is_complete": true, "processing_time": 1.23}
```

## 核心功能详解

### 1. BM25 关键词检索 + Redis 缓存

- 使用 `rank-bm25` 库构建 BM25Okapi 模型
- 对用户查询进行 Jieba 分词后计算与知识库所有问题的相似度
- Softmax 归一化后取最高分，超过阈值（默认 0.85）则命中
- 命中后自动缓存到 Redis，下次相同查询直接返回

### 2. BERT 意图识别

- 基于 `bert-base-chinese` 微调的二分类模型
- 将用户查询分为「通用知识」（LLM 直接回答）和「专业咨询」（触发 RAG 检索）
- 训练数据：5000 条标注样本（`classify_data/model_generic_5000.json`）

### 3. 多策略检索增强

| 策略 | 适用场景 | 原理 |
|------|---------|------|
| **直接检索** | 查询意图明确 | 直接用原查询进行混合检索 |
| **HyDE 检索** | 查询抽象/开放 | LLM 先生成假设答案，用假设答案检索 |
| **子查询检索** | 多实体/多方面查询 | LLM 拆分复杂查询为多个子查询，分别检索后合并去重 |
| **回溯问题检索** | 查询过于复杂 | LLM 将复杂问题简化为基础问题后检索 |

策略选择由 LLM（Qwen-Plus）根据查询特征自动决策。

### 4. Milvus 混合检索 + 重排序

- **稠密向量**（Dense）：BGE-M3 1024 维，负责语义级匹配
- **稀疏向量**（Sparse）：BGE-M3 词权重，负责关键词级精准匹配
- **加权融合**：稠密:稀疏 = 1.0:0.7 加权求和
- **重排序**：BGE-Reranker-Large Cross-Encoder 对候选父文档二次排序
- **分层存储**：父块（1200 字，保留上下文）+ 子块（300 字，精准匹配），子块检索后回溯父块

### 5. 对话历史管理

- 基于 MySQL `conversations` 表存储对话记录
- 自动保留最近 5 轮对话作为上下文
- 新增对话时自动淘汰超出 5 轮的历史记录
- 支持按 session_id 查询和清除历史

### 6. 流式输出

- WebSocket 长连接逐 Token 推送，实现打字机效果
- 兼容 OpenAI 格式的 DashScope API 流式响应
- 前端使用 marked.js 实时渲染 Markdown

### 7. RAG 质量评估

运行 RAGAS 评估脚本对系统进行定量评估：

```bash
cd rag_qa/rag_assessment
python rag_as.py
```

评估指标：
- **忠实度（Faithfulness）**：答案是否源自检索到的上下文
- **答案相关性（Answer Relevancy）**：答案与问题的匹配程度
- **上下文精确率（Context Precision）**：检索到的文档是否相关
- **上下文召回率（Context Recall）**：相关文档是否被检索到

## 支持的学科类别

默认支持 5 大学科（可在 `config.ini` 中扩展）：

- `ai` — 人工智能 / 机器学习
- `java` — Java 开发
- `test` — 软件测试
- `ops` — 运维
- `bigdata` — 大数据

通过 `/api/query` 的 `source_filter` 参数或前端下拉框进行学科过滤。

## 前端功能

- 💬 实时聊天界面，支持 Markdown 渲染
- 🌓 暗色模式 / 亮色模式切换
- 📱 响应式设计（适配移动端）
- 🔄 历史会话管理（新建 / 查看 / 清除）
- 🔍 历史会话搜索
- ⌨️ Enter 发送、Shift+Enter 换行
- ⚡ 快捷提问按钮

## 项目版本演进

| 版本 | 文件 | 特点 |
|------|------|------|
| **V1** | `old_main.py` + `rag_system.py` | 基础 MySQL + RAG 融合，同步返回完整答案 |
| **V2** | `new_main.py` + `new_rag_system.py` | 新增对话历史（MySQL 5 轮）、流式输出（WebSocket Token 推送） |

## 注意事项

1. **模型文件未包含在仓库中**：`rag_qa/models/` 目录下的 BGE-M3、BGE-Reranker、BERT 等模型文件需自行下载并放置在对应路径
2. **数据文件未包含在仓库中**：`rag_qa/data/` 目录下的学科文档需自行准备
3. **API Key 需自行配置**：`config.ini` 中的 DashScope API Key 为占位值，请替换为实际密钥
4. **Milvus 需提前启动**：确保 Milvus 服务已运行，并创建了配置中指定的 database
5. **Python 缓存文件**：项目中的 `__pycache__/` 目录已在 `.gitignore` 中排除
