# 网络检索功能说明

## 功能概述

Agent 现在具备互联网检索能力，可以通过网络搜索来判断一个名称是 **benchmark（数据集）** 还是 **模型**。这大大提高了判断的准确性和泛化性。

## 工作原理

1. **上下文分析**（优先）：首先基于查询模式和上下文关键词进行判断
2. **网络搜索**（回退）：如果上下文分析无法确定，则进行网络搜索
3. **结果缓存**：搜索结果会被缓存 7 天，避免重复查询

## 安装依赖

### 方案 1：使用 Google 搜索（推荐）

```bash
pip install googlesearch-python
```

### 方案 2：使用 DuckDuckGo 搜索（备用）

```bash
pip install requests beautifulsoup4
```

### 方案 3：安装所有依赖

```bash
pip install googlesearch-python requests beautifulsoup4
```

## 使用示例

### 示例 1：明确上下文
- **查询**："RT-1 上的 SOTA"
- **判断**：上下文分析 → RT-1 是数据集（匹配"X 上的 SOTA"模式）
- **网络搜索**：不需要

### 示例 2：模糊上下文
- **查询**："RT-1 的性能"
- **判断**：上下文分析无法确定
- **网络搜索**：搜索 "RT-1 dataset benchmark" 和 "RT-1 model architecture"
- **结果**：根据搜索结果判断 RT-1 是模型还是数据集

### 示例 3：新名称
- **查询**："NewModel-2024 上的最新结果"
- **判断**：不在已知数据集列表中
- **网络搜索**：自动搜索判断 NewModel-2024 的类型
- **结果**：根据搜索结果判断

## 缓存机制

- **缓存位置**：`papers/web_search_cache/name_type_cache.json`
- **缓存时长**：7 天
- **自动管理**：首次查询后自动缓存，后续查询直接使用缓存

## 搜索策略

### Google 搜索（如果可用）
- 搜索查询：`"{name}" dataset benchmark` 和 `"{name}" model architecture`
- 分析 URL 和标题中的关键词
- 数据集指标：dataset, benchmark, evaluation, paperwithcode
- 模型指标：model, architecture, arxiv.org/abs, github.com

### DuckDuckGo 搜索（备用）
- 搜索查询：`{name} dataset OR benchmark OR model`
- 分析页面内容中的关键词频率
- 根据关键词出现次数判断

## 性能考虑

- **延迟**：网络搜索会增加 1-3 秒的延迟
- **缓存**：已查询的名称会使用缓存，几乎无延迟
- **限流**：Google 搜索会自动限流，避免请求过快

## 注意事项

1. **网络连接**：需要稳定的网络连接
2. **API 限制**：Google 搜索可能有频率限制
3. **隐私**：搜索查询会发送到外部服务
4. **准确性**：搜索结果仅供参考，最终判断仍以上下文分析为主

## 故障排除

### 问题：网络搜索不工作

**检查项**：
1. 是否安装了依赖：`pip list | grep googlesearch`
2. 网络连接是否正常
3. 查看终端日志中的错误信息

**解决方案**：
- 安装缺失的依赖
- 检查防火墙设置
- 如果 Google 搜索不可用，会自动回退到 DuckDuckGo

### 问题：搜索结果不准确

**原因**：
- 搜索结果可能包含噪声
- 某些名称可能既是数据集又是模型（如某些基准测试套件）

**解决方案**：
- 在查询中使用更明确的上下文（如"RT-1 数据集"）
- 手动清除缓存：删除 `papers/web_search_cache/name_type_cache.json`

## 未来改进

- [ ] 支持更多搜索引擎（Bing, SerpAPI 等）
- [ ] 使用 LLM 分析搜索结果摘要
- [ ] 支持自定义搜索 API key
- [ ] 更智能的缓存策略（基于置信度）



