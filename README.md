<div align="center">

<img src="https://pic.qiuguang.top/1_00000.7i0tjvunlu.webp" alt="SOTAgent Logo" width="400"/>


# Intelligent SOTA Model Search Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange.svg)](https://github.com/QiuGuangwww/SOTAgent)

> An intelligent Agent built on Google ADK that helps researchers quickly find the latest SOTA models on specified benchmarks.

</div>

---

## 📖 Overview

With the rapid growth in the number of AI conferences and papers, selecting appropriate baselines and finding the latest State-of-the-Art (SOTA) models has become increasingly challenging. After the Paperwithcode website became unavailable, finding current state-of-the-art models often requires significant time and effort.

**SOTAgent** is an intelligent search assistant that can:
- 🔍 **Intelligent Search**: Find the latest SOTA models on specified benchmarks through natural language queries
- 🌐 **Web Search**: Automatically determine whether a name is a dataset or model using internet search and LLM analysis
- 📊 **Multi-dimensional Filtering**: Precisely filter by training paradigm, data constraints, modality, and more
- 🎯 **Context Understanding**: Intelligently understand query intent, distinguishing between "latest" and "strongest" SOTA
- 🔄 **Multi-Agent Pipeline**: Optional multi-agent collaborative workflow for more reliable verification results

---

## ✨ Core Features

### 1. Natural Language SOTA Queries
Supports both Chinese and English natural language queries with automatic intent parsing:

```python
# Example queries
"Find the latest SOTA model on GOT-10k"
"Supervised SOTA on RT-1 dataset"
"Find the strongest SOTA model (best performance)"
"SOTA in VLA domain in the last 6 months"
```

### 2. Intelligent Name Type Recognition
Automatically determines whether a name is a dataset or model through web search and LLM analysis:

- ✅ **Context Analysis**: Intelligent judgment based on query patterns
- ✅ **Web Search**: Verify with the latest information
- ✅ **LLM Analysis**: Use large language models to understand search result content
- ✅ **Result Caching**: Avoid duplicate queries for improved efficiency

### 3. Multi-dimensional Filtering & Constraints
Supports rich filtering conditions:

- **Training Paradigms**: supervised, self-supervised, zero-shot, few-shot, etc.
- **Data Constraints**: no-extra-data, extra-data, pretrained, etc.
- **Modality Types**: RGB, RGBD, multimodal, etc.
- **Techniques**: TTA, ensemble, prompting, etc.
- **Resource Constraints**: realtime, lightweight, etc.

### 4. Strict/Relaxed Filtering Modes
- **Strict Mode**: Precisely match all constraint conditions
- **Relaxed Mode**: Automatically relax constraints if strict filtering yields no results

### 5. Performance-First Sorting
When users query for "strongest" or "best performance":
- Sort by performance metrics (rather than time)
- Automatically exclude lightweight models
- Return truly performance-optimal models

### 6. Multi-Agent Pipeline (Optional)
Provides more reliable trusted SOTA search:
- **Agent A (Scanner)**: Multi-source search (arXiv, Google Scholar)
- **Agent B (Extractor)**: PDF content extraction (with Vision Model support)
- **Agent C (Normalizer)**: Metric standardization
- **Agent D (Verifier)**: Conflict detection and verification

### 7. Beautiful Web Interface
- 🎨 Modern UI design with light/dark theme switching
- 💬 Full-width conversation window with optimized message bubbles
- ⚙️ Rich configuration options (filtering modes, Vision Model, etc.)
- 📱 Responsive design with mobile support

---

## 🚀 Quick Start

### Requirements

- Python 3.8+
- Google ADK
- Gemini API Key (or other supported LLM API keys)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/QiuGuangwww/SOTAgent.git
cd SOTAgent
```

#### 2. Install Dependencies

**Using uv (Recommended)**:
```bash
pip install uv
uv pip install -r requirements.txt
```

**Or using pip**:
```bash
pip install -r requirements.txt
```

**Install Web Search Functionality (Optional but Recommended)**:
```bash
pip install googlesearch-python requests beautifulsoup4
```

**Install Multi-Agent Pipeline Functionality (Optional)**:
```bash
pip install -r My_First_Agent/requirements_pipeline.txt
```

#### 3. Configure API Key

Create a `.env` file (in the project root or `My_First_Agent` directory):

```bash
# Method 1: Environment variable
export GEMINI_API_KEY=your_api_key_here

# Method 2: .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

Get Gemini API Key: https://aistudio.google.com/

#### 4. Launch the Application

**Web Interface (Recommended)**:
```bash
python app.py
```

The application will start at `http://localhost:50001` (port may vary, check terminal output).

**Command Line Interface**:
```bash
uv run adk web
```

---

## 📚 Usage Guide

### Basic Query Examples

#### 1. Find Latest SOTA
```
Find the latest SOTA model on GOT-10k
```

#### 2. Query with Constraints
```
Supervised SOTA on RT-1 dataset without extra data
```

#### 3. Find Best Performance
```
Find the best performing SOTA model on GOT-10k
```

#### 4. Time Window Query
```
SOTA in VLA domain in the last 180 days
```

#### 5. Name Type Query
```
Is RT-1 a dataset?
```

### Advanced Features

#### Filtering Mode Selection
- **Strict Mode**: Precisely match all constraints, suitable for cases requiring exact results
- **Relaxed Mode**: Automatically relax constraints if strict filtering yields no results, suitable for exploratory queries

#### Vision Model Enhancement (Beta)
When enabled, the Agent can use Vision Models (GPT-4V, Claude Vision, Gemini Vision) to process complex tables and charts:
- Suitable for scenarios requiring extraction of complex tables from PDFs
- Increases processing time and cost
- Automatically enabled when using "find in a trustworthy way..." in queries

#### Trusted SOTA Search
Use Multi-Agent Pipeline for multi-source verification:
```
Find SOTA on GOT-10k in a trustworthy way
```

---

## 🏗️ Project Structure

```
SOTAgent/
├── app.py                          # Gradio Web application main file
├── main.py                         # Command line entry point
├── pyproject.toml                  # Project configuration
├── requirements.txt                # Basic dependencies
├── My_First_Agent/
│   ├── agent.py                    # Core Agent definition and tool functions
│   ├── multi_agent_pipeline.py    # Multi-Agent Pipeline implementation
│   ├── pipeline_tools.py           # Pipeline tool interfaces
│   ├── vision_extractor.py         # Vision Model extractor
│   ├── requirements_pipeline.txt   # Pipeline dependencies
│   └── README_*.md                 # Feature documentation
├── papers/                         # Paper cache directory
│   ├── web_search_cache/           # Web search cache
│   └── [topic]/                    # Papers organized by topic
└── README.md                       # This file
```

---

## 🔧 Core Tool Functions

The Agent provides the following main tools:
<div align="center">

| Tool Function | Description |
|--------------|-------------|
| `search_papers` | Search for papers on a specific topic |
| `extract_info` | Extract detailed paper information |
| `find_papers_by_benchmark` | Find papers by benchmark |
| `get_latest_sota` | Find latest SOTA models (supports natural language) |
| `list_recent_papers` | List recent papers |
| `recent_by_nl` | Natural language query for recent papers |
| `list_common_benchmarks` | List common benchmarks |
| `list_common_benchmarks_with_sota` | List common benchmarks with their SOTA |
| `check_name_type` | Check if a name is a dataset or model |
| `run_trustworthy_sota_search` | Trusted SOTA search (Pipeline) |

</div>

---

## ⚙️ Configuration

### Environment Variables

- `GEMINI_API_KEY`: Gemini API key (required)

### Agent Configuration

In `My_First_Agent/agent.py`, you can modify:

```python
# Select the model to use
use_model = "gemini"  # Options: "gemini", "gpt-4o", "deepseek"

# Model configuration
if use_model == "gemini":
    model = LiteLlm(model="gemini/gemini-2.5-flash")
```

### Web Interface Configuration

In `app.py`, you can modify:
- Port number (default: automatically find available port)
- UI theme and styles
- Feature toggles

---

## 📝 Development

### Adding New Benchmarks

Add to the `_DATASET_PATTERNS` dictionary in `My_First_Agent/agent.py`:

```python
_DATASET_PATTERNS: Dict[str, List[str]] = {
    # ... existing entries
    "YourBenchmark": [r"\byourbenchmark\b", r"\byb\b"],
}
```

### Adding New Filter Conditions

Add patterns to the corresponding `_PATTERNS` dictionary, for example:

```python
_DATA_REGIME_PATTERNS: Dict[str, List[str]] = {
    # ... existing entries
    "your-constraint": [r"\byour\s+constraint\b"],
}
```

### Extending Web Search Functionality

Modify the `_web_search_name_type()` function to:
- Add support for more search engines
- Improve LLM analysis prompts
- Optimize caching strategies

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Google ADK](https://github.com/google/adk) - Agent development framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified interface for multiple LLMs
- [Gradio](https://github.com/gradio-app/gradio) - Web interface framework
- [arXiv](https://arxiv.org/) - Paper data source

---

## 📧 Contact

For questions or suggestions, please contact:

- 📮 Email: [qiuguang738@gmail.com](mailto:qiuguang738@gmail.com)
- 🐛 Issues: [GitHub Issues](https://github.com/QiuGuangwww/SOTAgent/issues)

---

<div align="center">

**⭐ If this project helps you, please give it a Star!**

Last update: 2025.11.25

</div>


