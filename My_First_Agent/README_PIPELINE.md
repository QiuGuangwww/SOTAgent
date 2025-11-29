# Multi-Agent Pipeline MVP

## 概述

这是一个可信 SOTA 追踪的 Multi-Agent Pipeline MVP 版本，实现了论文中描述的四个 Agent：

- **Agent A (Scanner)**: 多源搜索（arXiv + Google Scholar）
- **Agent B (Extractor)**: PDF 文本和表格提取
- **Agent C (Normalizer)**: 指标标准化和转换
- **Agent D (Verifier)**: 冲突检测和验证

## 安装依赖

```bash
pip install -r requirements_pipeline.txt
```

主要依赖：
- `PyMuPDF` (fitz): PDF 文本提取
- `pdfplumber`: 表格提取
- `scholarly`: Google Scholar 搜索（可选）
- `requests`: PDF 下载

## 使用方法

### 1. 直接使用 Pipeline

```python
from multi_agent_pipeline import run_sota_pipeline
import asyncio

async def main():
    result = await run_sota_pipeline("GOT-10k tracking SOTA", max_papers=5)
    print(result)

asyncio.run(main())
```

### 2. 通过 Agent 调用

Pipeline 已集成到现有的 Agent 系统中，可以通过自然语言调用：

```
用户: "用可信的方式找 GOT-10k 上最强的 SOTA 模型"
Agent: 会调用 run_trustworthy_sota_search 函数
```

### 3. 测试

```bash
python test_pipeline.py
```

## Pipeline 流程

```
用户查询
    ↓
[Agent A: Scanner]
    ├─ arXiv 搜索
    └─ Google Scholar 搜索
    ↓
[Agent B: Extractor]
    ├─ 下载 PDF
    ├─ 提取文本
    ├─ 提取表格
    └─ 提取指标
    ↓
[Agent C: Normalizer]
    ├─ 标准化指标名称
    ├─ 统一单位
    └─ 指标转换（如 Error Rate → Accuracy）
    ↓
[Agent D: Verifier]
    ├─ 对比多源数据
    ├─ 检测冲突
    └─ 计算置信度
    ↓
最终结果
```

## MVP 限制

当前 MVP 版本的已知限制：

1. **PDF 提取**:
   - ✅ 支持文本提取
   - ✅ 支持简单表格提取
   - ❌ 不支持复杂表格（合并单元格、跨页表格）
   - ❌ 不支持图表 OCR（需要 Vision Model）

2. **指标提取**:
   - ✅ 基础正则表达式提取
   - ❌ 不支持上下文理解（可能误提取）

3. **标准化**:
   - ✅ 基础指标转换规则
   - ✅ 数据集别名映射
   - ❌ 不完整的转换规则库

4. **验证**:
   - ✅ 基础冲突检测
   - ❌ 不包含置信度评分算法

## 下一步改进

1. **增强 PDF 提取**:
   - 集成 Vision Model（GPT-4V/Claude Vision）处理复杂表格
   - 添加图表 OCR

2. **改进指标提取**:
   - 使用 LLM 进行上下文理解
   - 更精确的指标识别

3. **完善标准化**:
   - 扩展指标转换规则库
   - 添加更多数据集别名

4. **增强验证**:
   - 实现更复杂的置信度评分
   - 添加来源可信度评估

## 文件结构

```
My_First_Agent/
├── multi_agent_pipeline.py    # Pipeline 核心实现
├── pipeline_tools.py           # Agent 集成工具
├── test_pipeline.py            # 测试脚本
├── requirements_pipeline.txt   # 依赖列表
└── README_PIPELINE.md          # 本文档
```

## 注意事项

1. **Google Scholar 搜索**: `scholarly` 库可能不稳定，如果失败会自动跳过
2. **PDF 下载**: 某些 PDF 可能无法下载（需要权限、链接失效等）
3. **处理时间**: Pipeline 处理较慢（每篇论文需要下载和解析 PDF），建议 `max_papers` 设置为 3-5
4. **存储空间**: PDF 会缓存在 `papers/extracted/` 目录

## 故障排除

### 问题: `scholarly` 导入失败
**解决**: 这是可选的，Pipeline 会自动跳过 Google Scholar 搜索

### 问题: PDF 下载失败
**解决**: 检查网络连接，某些 PDF 链接可能需要特殊权限

### 问题: 表格提取不准确
**解决**: MVP 版本只支持简单表格，复杂表格需要 Vision Model（未来版本）


