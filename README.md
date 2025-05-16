# TechReq: Multi-Agent Technical Specification Generation

This project implements a multi-agent system for generating comprehensive technical specifications for hardware product development using Large Language Models (LLMs). The system simulates an organization of diverse domain experts collaborating to create detailed engineering requirements from unstructured customer needs.

## Features

- Multi-agent collaboration framework
- Support for multiple LLM models:
  - GPT-4
  - Claude
  - Llama
- Pre-processed customer needs for various products
- Comprehensive technical specification generation
- Domain-specific requirement analysis

## Getting Started

### Prerequisites

It's recommended to create a new conda environment:

```bash
conda create -n your_env_name python=3.9
conda activate your_env_name
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lindapu-1/TechReq.git
cd TechReq
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

The script supports various command-line arguments for flexible execution:

```bash
python main.py [options]
```

Available options:
- `--model`: Specify which LLM to use:
  - `gpt4o` (default): GPT-4
  - `claude3`: Claude-3-Opus (version 20240229)
  - `claude37`: Claude-3-Sonnet (version 20250219)
  - `llama31`: Llama 3.1
  - `gpt35`: GPT-3.5 Turbo
- `--needs-file`: Path to existing customer needs JSON file (e.g., `cus_needs/helmets.json`), if you want to skip the first stage
- `--domain-file`: Path to existing domain needs JSON file, if you want to skip the second stage and start from the round robin chat between expert agents
- `--final-json-filename`: Path to existing final metrics JSON file, if you just want to test the score from LLM-as-judge with existing final metrics.
- `--result-folder`: Path of the result folder if you want to specify a custom result folder, otherwise the result will be created automatically in the name of the current time.

Example commands:
```bash
# Run with default model (gpt4o)
python main.py --needs-file cus_needs/helmets.json

# Run with Claude-3-Opus
python main.py --model claude3 --needs-file cus_needs/desktop_heaters_needs.json

# Run with custom result folder using Llama 3.1
python main.py --model llama31 --needs-file cus_needs/helmets.json --result-folder custom_results
```

We have pre-processed customer needs files for five products in `cus_needs/` where you can use for testing:
- `folding_knives_needs.json`
- `childrens_multi-purpose_bike_helmets_needs.json`
- `Mini_Travel_Hair_Dryers_needs.json`
- `desktop_heaters_needs.json`
- `helmets.json`
- `suspensionall.json`

## Project Structure

```
.
├── main.py              # Main execution script
├── tools.py            # Utility functions and tools
├── em.py               # Engineering management functions
├── cus_needs/          # Pre-processed customer needs
├── domain_needs/       # Domain-specific requirements
├── results/            # Automatically generated results folder
└── report/            # Analysis reports
```

## Note

The Elicitron API, responsible for generating initial customer needs, is not publicly accessible. For testing and development purposes, it is highly recommended to utilize the pre-processed customer needs files located in the `cus_needs` folder. **This limitation does not compromise the project's objectives or primary achievements.** 