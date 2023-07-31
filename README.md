# LLM App Stack

*aka Emerging Architectures for LLM Applications*

![llm-app-stack](https://github.com/a16z-infra/llm-app-stack/assets/26883865/92734642-9651-4aaa-a803-79a6cf5414ef)

This is a list of available tools, projects, and vendors at each layer of the LLM app stack. 

Our [original article](https://a16z.com/2023/06/20/emerging-architectures-for-llm-applications/) included only the most popular tools based on user interviews prior to publication (June 20th 2023). This repo, instead, attempts to be comprehensive, covering all options in each category. If you see anything missing from list, or miscategorized, please open a PR!


## Table of Contents
1. [Data Pipelines](#data-pipelines)
2. [Embedding Model](#embedding-model)
3. [Vector Database](#vector-database)
4. [Playground](#playground)
5. [Orchestration](#orchestration)
6. [APIs / Plugins](#apis--plugins)
7. [LLM Cache](#llm-cache)
8. [Logging / LLMops](#logging--llmops)
9. [Validation](#validation)
10. [App Hosting](#app-hosting)
11. [LLM APIs (proprietary)](#llm-apis-proprietary)
12. [LLM APIs (open source)](#llm-apis-open-source)
13. [Cloud Providers](#cloud-providers)
14. [Opinionated Clouds](#opinionated-clouds)


## Comprehensive tool list

### Data Pipelines
| Name (site) | Description | Github | Pip Installs |
| --- | --- | --- | --- |
| [Databricks](https://databricks.com/) | A unified set of tools for building, deploying, sharing, and maintaining enterprise-grade data solutions at scale | ![GitHub Repo stars](https://img.shields.io/github/stars/apache/spark?style=social) | <a href=https://pypi.org/project/pyspark><img src="https://img.shields.io/pypi/dw/pyspark" width=150/></a> |
| [Airflow](https://airflow.apache.org/) | A platform to programmatically author, schedule, and monitor data pipelines & workflows | ![GitHub Repo stars](https://img.shields.io/github/stars/apache/airflow?style=social) | <a href=https://pypi.org/project/apache-airflow><img src="https://img.shields.io/pypi/dw/apache-airflow" width=150/></a> |
| [Unstructured.io](https://unstructured.io/) | Open-source components for pre-processing text documents such as PDFs, HTML and Word Documents for usage with LLM apps | ![Unstructured-IO/unstructured](https://img.shields.io/github/stars/Unstructured-IO/unstructured?style=social) | <a href=https://pypi.org/project/unstructured><img src="https://img.shields.io/pypi/dw/unstructured" width=150/></a> |
| [Fivetran](https://www.fivetran.com/) | A trusted platform that extracts, loads, and transforms data from various sources for analytics or operations [2][7] | N/A | <a href=https://pypi.org/project/fivetran><img src="https://img.shields.io/pypi/dw/fivetran" width=150/></a> |


### Embedding Model
| Name (site) | Description | Github | Pip Installs |
| --- | --- | --- | --- |
| [Open AI  Ada 2](https://platform.openai.com/docs/guides/embeddings) | A powerful text embedding model for capturing semantic relationships in text | n/a | <a href=https://pypi.org/project/openai><img src="https://img.shields.io/pypi/dw/openai" width=150/></a> |
| [Cohere AI](https://docs.cohere.com/docs/embeddings) | Generate text embeddings with large language models for semantic search, topic clustering, and more | ![cohere-ai/notebooks](https://img.shields.io/github/stars/cohere-ai/notebooks?style=social)| <a href=https://pypi.org/project/cohere><img src="https://img.shields.io/pypi/dw/cohere" width=150/></a> |
| [Sentence Transformers](https://huggingface.co/) | A Python framework for state-of-the-art sentence, text, and image embeddings | ![UKPLab/sentence-transformers](https://img.shields.io/github/stars/UKPLab/sentence-transformers?style=social) | <a href=https://pypi.org/project/sentence-transformers><img src="https://img.shields.io/pypi/dw/sentence-transformers" width=150/></a> |



### Vector Database
| Name (site) | Description | Github | Pip Installs |
| --- | --- | --- | --- |
| [Pinecone](https://pinecone.io/) | A managed, cloud-native vector database with a simple API for high-performance AI applications | n/a | <a href=https://pypi.org/project/pinecone-client><img src="https://img.shields.io/pypi/dw/pinecone-client" width=150/></a> |
| [Weaviate](https://weaviate.io/) | An open-source vector database that stores both objects and vectors, allowing for combining vector search with structured filtering | ![semi-technologies/weaviate](https://img.shields.io/github/stars/semi-technologies/weaviate?style=social) | <a href=https://pypi.org/project/weaviate-client><img src="https://img.shields.io/pypi/dw/weaviate-client" width=150/></a> |
| [ChromaDB](https://trychroma.com/) | An AI-native open-source embedding database platform for developers to add state and memory to their AI-enabled applications | ![chroma-core/chroma](https://img.shields.io/github/stars/chroma-core/chroma?style=social) | <a href=https://pypi.org/project/chromadb><img src="https://img.shields.io/pypi/dw/chromadb" width=150/></a> |
| [pgvector](https://github.com/pgvector/pgvector) | Open-source vector similarity search for Postgres, allowing for exact and approximate nearest neighbor search | ![pgvector/pgvector](https://img.shields.io/github/stars/pgvector/pgvector?style=social) | <a href=https://pypi.org/project/pgvector><img src="https://img.shields.io/pypi/dw/pgvector" width=150/></a> |


### Playground
| Name (site) | Description | Github | Pip Installs |
| --- | --- | --- | --- |
| [OpenAI Playground](https://platform.openai.com/) | A web-based platform for experimenting with various machine-learning models developed by OpenAI | N/A | N/A |
| [nat.dev](https://nat.dev) | A platform that allows users to test prompts with multiple language models, both commercial and open-source, and compare their performance | ![nat/openplayground](https://img.shields.io/github/stars/nat/openplayground?style=social) | <a href=https://pypi.org/project/openplayground><img src="https://img.shields.io/pypi/dw/openplayground" width=150/></a> |
| [Humanloop](https://humanloop.com/) | A platform that helps developers build high-performing applications on top of large language models like GPT-3, with tools for experimenting, collecting data, and fine-tuning models | ![humanloop/humanloop-tutorial-python](https://img.shields.io/github/stars/humanloop/humanloop-tutorial-python?style=social) | <a href=https://pypi.org/project/humanloop><img src="https://img.shields.io/pypi/dw/humanloop" width=150/></a> |


### Orchestration
| Name (site) | Description | Github | Pip Installs |
| --- | --- | --- | --- |
| [Langchain](https://langchain.com/) | An open-source library that provides developers with the tools to build applications powered by large language models (LLMs) | ![langchain-ai/langchain](https://img.shields.io/github/stars/langchain-ai/langchain?style=social) | <a href=https://pypi.org/project/langchain><img src="https://img.shields.io/pypi/dw/langchain" width=150/></a> |
| [LlamaIndex](https://llamaindex.ai/) | A data framework for LLM applications to ingest, structure, and access private or domain-specific data | ![jerryjliu/llama_index](https://img.shields.io/github/stars/jerryjliu/llama_index?style=social) | <a href=https://pypi.org/project/llama-index><img src="https://img.shields.io/pypi/dw/llama-index" width=150/></a> |
| [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel) | A lightweight open-source orchestration SDK that lets you easily mix-and-match AI prompts with conventional programming languages like C and Python [1] | ![microsoft/semantic-kernel](https://img.shields.io/github/stars/microsoft/semantic-kernel?style=social) | <a href=https://pypi.org/project/semantic-kernel><img src="https://img.shields.io/pypi/dw/semantic-kernel" width=150/></a> |
| [Vercel AI SDK](https://sdk.vercel.ai/docs) | An open-source library designed to help developers build conversational streaming user interfaces in JavaScript and TypeScript [4] | ![vercel/ai](https://img.shields.io/github/stars/vercel-labs/ai?style=social) | <a href=https://pypi.org/project/vercel-ai-sdk><img src="https://img.shields.io/npm/dw/ai" width=150/></a>(node/npm)|
| [Vectara AI](https://vectara.com/) | A GenAI conversational search and discovery platform that allows businesses to have intelligent conversations utilizing their own data | ![vectara/vectara-ingest](https://img.shields.io/github/stars/vectara/vectara-ingest?style=social) | N/A |
| [ChatGPT](https://chat.openai.com) | An AI chatbot that uses natural language processing to create humanlike conversational dialogue | | |

### APIs / Plugins
| Name (site) | Description | Github | Pip Installs |
| --- | --- | --- | --- |
| [Serp API](https://serpapi.com/) | A real-time API to access Google search results, handling proxies, captchas, and parsing structured data | ![serpapi/google-search-results](https://img.shields.io/github/stars/serpapi/google-search-results-python?style=social) | <a href=https://pypi.org/project/google-search-results><img src="https://img.shields.io/pypi/dw/google-search-results" width=150/></a> |
| [Wolfram Alpha API](https://wolframalpha.com/) | A web-based API providing computational and presentation capabilities for integration into various applications | N/A | <a href=https://pypi.org/project/wolframalpha><img src="https://img.shields.io/pypi/dw/wolframalpha" width=150/></a> |
| [Zapier API AI Plugin](https://zapier.com/) | A plugin that allows you to connect 5,000+ apps and interact with them directly inside ChatGPT | N/A | N/A |

### LLM Cache
| Name (site) | Description | Github | Pip Installs |
| --- | --- | --- | --- |
| [Redis](https://redis.io/) | An in-memory data structure store used as a database, cache, message broker, and streaming engine | ![redis/redis](https://img.shields.io/github/stars/redis/redis?style=social) | <a href=https://pypi.org/project/redis/><img src="https://img.shields.io/pypi/dw/redis" width=150/></a> |
| [SQLite](https://sqlite.org/) | A self-contained, serverless, zero-configuration, transactional SQL database engine | ![sqlite/sqlite](https://img.shields.io/github/stars/sqlite/sqlite?style=social) | <a href=https://pypi.org/project/pysqlite3/><img src="https://img.shields.io/pypi/dw/pysqlite3" width=150/></a> |
| [GPTCache](https://github.com/zilliztech/GPTCache) | An open-source tool designed to improve the efficiency and speed of GPT-based applications by implementing a cache to store the responses | ![zilliztech/GPTCache](https://img.shields.io/github/stars/zilliztech/GPTCache?style=social) | N/A |

### Logging / LLMops
| Name (site) | Description | Github | Pip Installs |
| --- | --- | --- | --- |
| [Weights & Biases](https://wandb.ai) | A developer-first MLOps platform for streamlining ML workflows | ![wandb/wandb](https://img.shields.io/github/stars/wandb/wandb?style=social) | <a href=https://pypi.org/project/wandb><img src="https://img.shields.io/pypi/dw/wandb" width=150/></a> |
| [MLflow](https://mlflow.org/docs/latest/llm-tracking.html) | A platform to streamline machine learning development, including tracking experiments, packaging code, and sharing models | ![mlflow/mlflow](https://img.shields.io/github/stars/mlflow/mlflow?style=social) | <a href=https://pypi.org/project/mlflow><img src="https://img.shields.io/pypi/dw/mlflow" width=150/></a> |
| [PromptLayer](https://promptlayer.com/) | A platform for tracking, managing, and sharing GPT prompt engineering | ![MagnivOrg/prompt-layer-library](https://img.shields.io/github/stars/MagnivOrg/prompt-layer-library?style=social) | <a href=https://pypi.org/project/promptlayer><img src="https://img.shields.io/pypi/dw/promptlayer" width=150/></a> |
| [Helicone](https://helicone.ai/) | An open-source observability platform for Language Learning Models (LLMs) | ![Helicone/helicone](https://img.shields.io/github/stars/Helicone/helicone?style=social) | <a href=https://pypi.org/project/helicone><img src="https://img.shields.io/pypi/dw/helicone" width=150/></a> |
| [Arize AI](https://arize.com/) | A Machine Learning Observability platform that helps ML practitioners successfully take models from research to production, with ease| ![Arize-ai](https://img.shields.io/github/stars/Arize-ai?style=social) | <a href=https://pypi.org/project/arize><img src="https://img.shields.io/pypi/dw/arize" width=150/></a> |

### Validation
| Name (site) | Description | Github | Pip Installs |
| --- | --- | --- | --- |
| [Guardrails AI](https://shreyar.github.io/guardrails/) | An open-source Python package for specifying structure and type, validating and correcting the outputs of large language models (LLMs) | ![ShreyaR/guardrails](https://img.shields.io/github/stars/ShreyaR/guardrails?style=social) | <a href=https://pypi.org/project/guardrails-ai><img src="https://img.shields.io/pypi/dw/guardrails-ai" width=150/></a> |
| [Rebuff](https://github.com/woop/rebuff) | An open-source framework designed to detect and protect against prompt injection attacks in Language Learning Model (LLM) applications | ![woop/rebuff](https://img.shields.io/github/stars/woop/rebuff?style=social) | <a href=https://pypi.org/project/rebuff><img src="https://img.shields.io/pypi/dw/rebuff" width=150/></a> |
| [Microsoft Guidance](https://github.com/microsoft/guidance) | A guidance language for controlling large language models, providing a simple and comprehensive syntax for architecting complex LLM workflows | ![microsoft/guidance](https://img.shields.io/github/stars/microsoft/guidance?style=social) | <a href=https://pypi.org/project/guidance><img src="https://img.shields.io/pypi/dw/guidance" width=150/></a> |
| [LMQL](https://lmql.ai/) | An open-source programming language and platform for language model interaction, designed to make working with language models like OpenAI more expressive and accessible | ![eth-sri/lmql](https://img.shields.io/github/stars/eth-sri/lmql?style=social) | <a href=https://pypi.org/project/lmql><img src="https://img.shields.io/pypi/dw/lmql" width=150/></a> |

### App Hosting
- Steamship (https://steamship.com/)
- Netlify (https://www.netlify.com/)
- Vercel (https://vercel.com/)
- Streamlit (https://streamlit.io/)
- Modal (https://modal.com)

### LLM APIs (proprietary)
- OpenAI (https://openai.com/)
- Anthropic (https://anthropic.com/)

### LLM APIs (open source)
- HuggingFace (https://huggingface.co/)
- Replicate (https://replicate.com/)

### Cloud Providers
- AWS (https://aws.amazon.com/)
- GCP (https://cloud.google.com/)
- Azure (https://azure.microsoft.com/en-us)
- Coreweave (https://coreweave.com/)

### Opinionated Clouds
- Databricks (https://databricks.com/)
- Anyscale (https://anyscale.com/)
- Mosaic (https://mosaicml.com)
- Modal (https://modal.com)
- Runpod (https://runpod.io/)
- OctoML (https://octoml.ai/)