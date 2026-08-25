# SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search

[![Paper](https://img.shields.io/badge/arXiv-2507.15245-b31b1b.svg)](https://arxiv.org/abs/2507.15245)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-SPARBench-ffbd21.svg)](https://huggingface.co/datasets/XiaofengAlg/SPARBench)
[![Cache](https://img.shields.io/badge/Hugging%20Face-SPAR%20Cache-ffbd21.svg)](https://huggingface.co/datasets/XiaofengAlg/SPAR-arxiv-cache)
[![License: MIT](https://img.shields.io/badge/License-MIT-2563eb.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-0f766e.svg)](CITATION.cff)

[中文说明](README_ZH.md)

SPAR is a multi-agent scholarly retrieval framework that combines RefChain-based
query decomposition, query evolution, citation-aware exploration, and
re-ranking. The accompanying **SPARBench** dataset provides expert-annotated
relevance labels for systematic evaluation.

> If you use SPAR or SPARBench, please cite the
> [SPAR paper](https://arxiv.org/abs/2507.15245). A machine-readable citation is
> available in [CITATION.cff](CITATION.cff).

![overview](./figs/graph_example.png)

## 🚀 Quick Start

### Requirements

SPAR supports Python 3.9 and newer.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Basic Configuration

1. **Configure API Keys**
   - Set provider credentials with `OPENAI_API_KEY`, `GOOGLE_SERPER_KEY`, and
     the optional `S2_API_KEY` environment variables. Never commit API keys.
   - Edit [`global_config.py`](global_config.py) to set non-secret search parameters
   - For local models, refer to [`local_request_v2.py`](local_request_v2.py) to configure `MODEL_CONFIGS`

2. **Launch Web Interface**
   ```bash
   python3 demo_app_with_front.py
   ```
   ![demo](./figs/search_demo.jpg)

   search result details can be found: [here](./figs/search_results_2025-07-22.json)

3. **Use Service Interface**
   ```bash
   python3 run_spr_agent.py $benchname
   ```
   Supported `benchname`: `OwnBenchmark` | `AutoScholarQuery`

##  Project Structure

| File | Description |
|------|-------------|
| [`search_engine.py`](search_engine.py) |  Main entry point for retrieval system |
| [`pipeline_spar.py`](pipeline_spar.py) |  Complete SPAR processing pipeline |
| [`search_node.py`](search_node.py) |  Specific functionality implementation for pipeline |
| [`rerank.py`](rerank.py) |  Result re-ranking module |
| [`global_config.py`](global_config.py) |  Global configuration file |
| [`demo_app_with_front.py`](demo_app_with_front.py) | Visual frontend application |

## 🔧 Advanced Configuration

### Local Database Acceleration (Optional)

A validated public snapshot of the scholarly metadata cache is available from
the [SPAR arXiv cache dataset](https://huggingface.co/datasets/XiaofengAlg/SPAR-arxiv-cache).
The compressed download is about 403 MB and expands to a 2.16 GB SQLite file
with 112,581 records.

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
and Zstandard, then download and unpack the cache:

```bash
pip install -U huggingface_hub
# macOS: brew install zstd
# Ubuntu/Debian: sudo apt-get install zstd

mkdir -p database
hf download XiaofengAlg/SPAR-arxiv-cache arxiv_data.db.zst \
  --type dataset \
  --revision 9d09ce50c5dc7a3c8875e92fa601ae93acde6429 \
  --local-dir database
zstd -d database/arxiv_data.db.zst -o database/arxiv_data.db
```

SPAR uses `./database/arxiv_data.db` by default. Set `SPAR_DB_PATH` to use a
different location. If no database exists, SPAR creates an empty local cache
and falls back to live scholarly APIs; downloading the snapshot is optional.
The pinned revision above keeps the published artifact reproducible. The cache
is a best-effort 2025 snapshot; see the dataset card for integrity, provenance,
and licensing details.

### Graphical Visualization (Optional)

Install Graphviz to generate tree diagrams of the retrieval process:

```bash
# Ubuntu/Debian
sudo apt-get install graphviz
pip install graphviz

# macOS
brew install graphviz
pip install graphviz

# Windows
# 1. Download and install Graphviz: https://graphviz.org/download/
# 2. pip install graphviz
```

Preview:
![graph tree](./figs/graph_example.png)

## 📈 Experimental Results

![main result](./figs/spar_main_result.png)

##  Output Description

- Retrieval results saved to: `./figs/search_results_2025-07-22.json`
- Visualization charts saved in `./figs/` directory

##  Features

### 🎯 Advanced Search Mode
- **Query Rewriting**: Automatic query expansion and refinement
- **Intent Analysis**: Understanding search intent for better results
- **Reference Search**: Follow citation networks for comprehensive coverage
- **Advanced Re-ranking**: Multi-layer relevance scoring

### ⚡ Simple Search Mode
- **Multi-source Search**: ArXiv, OpenAlex, PubMed integration
- **Basic Re-ranking**: Fast relevance scoring
- **Batch Processing**: Efficient parallel processing

### 🎨 Web Interface
- **Interactive UI**: User-friendly search interface
- **Real-time Results**: Live search progress and results
- **Export Options**: JSON export for further analysis
- **Search Tree Visualization**: Visual representation of search process

## 📖 Citation

If you use SPAR, SPARBench, or results produced with the system, please cite
the accompanying paper:

```bibtex
@misc{shi2025sparscholarpaperretrieval,
      title={SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search},
      author={Xiaofeng Shi and Yuduo Li and Qian Kou and Longbin Yu and Jinxin Xie and Hua Zhou},
      year={2025},
      eprint={2507.15245},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2507.15245},
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributing

Issues and Pull Requests are welcome to help improve the SPAR system!

##  Troubleshooting

### Common Issues

1. **API Quota Exceeded**: Ensure you have sufficient API quota for LLM calls
2. **Slow Performance**: Consider using local database acceleration
3. **Network Issues**: Check network connectivity for external API calls

### Performance Tips

- Test on small datasets first to evaluate system performance
- Use local models when possible to reduce API costs
- Enable database caching for frequently accessed papers

---

> **Note**: Ensure you have sufficient API quota for Large Language Model calls. It's recommended to test the system performance on small-scale data first.
