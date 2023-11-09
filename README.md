# LLM App Stack

*aka Emerging Architectures for LLM Applications*

![2657 Emerging LLM App Stack R2 Clean](https://github.com/a16z-infra/llm-app-stack/assets/26883865/9363af72-9d44-4c62-ad90-007234b5b791)

This is a list of available tools, projects, and vendors at each layer of the LLM app stack. 

Our [original article](https://a16z.com/2023/06/20/emerging-architectures-for-llm-applications/) included only the most popular options, based on user interviews. This repo is meant to be more comprehensive, covering all available options in each category. We probably still missed some important projects, so please open a PR if you see anything missing.

We also included [Perplexity and ChatGPT prompts](#formatting-prompt-templates) to make searching and markdown table formatting easier.


## Table of Contents

1. [Data Pipelines](#data-pipelines)
2. [Embedding Model](#embedding-model)
3. [Vector Database](#vector-database)
4. [Playground](#playground)
5. [Orchestrators](#orchestrators)
6. [APIs / Plugins](#apis--plugins)
7. [LLM Caches](#llm-cache)
8. [Logging / Monitoring / Eval](#logging--monitoring--eval)
9. [Validators](#validators)
10. [LLM APIs (proprietary)](#llm-apis-proprietary)
11. [LLM APIs (open source)](#llm-apis-open-source)
12. [App Hosting Platforms](#app-hosting-platforms)
13. [Cloud Providers](#cloud-providers)
14. [Opinionated Clouds](#opinionated-clouds)




## Project List

### Data Pipelines
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [Databricks](https://databricks.com/) | A unified data platform for building, deploying, and maintaining enterprise data solutions, including products (like MosaicML and MLflow) purpose-built for AI | <a href=https://github.com/apache/spark><img src="https://img.shields.io/github/stars/apache/spark?style=social" width=90/></a> | <a href=https://pypi.org/project/pyspark><img src="https://img.shields.io/pypi/dw/pyspark" width=150/></a> |
| [Airflow](https://airflow.apache.org/) | A data pipeline framework to programmatically author, schedule, and monitor data pipelines and workflows, including for LLMs | <a href=https://github.com/apache/airflow><img src="https://img.shields.io/github/stars/apache/airflow?style=social" width=90/></a> | <a href=https://pypi.org/project/apache-airflow><img src="https://img.shields.io/pypi/dw/apache-airflow" width=150/></a> |
| [Unstructured.io](https://unstructured.io/) | Open-source components for pre-processing documents such as PDFs, HTML, and Word Documents for usage with LLM apps | <a href=https://github.com/Unstructured-IO/unstructured><img src="https://img.shields.io/github/stars/Unstructured-IO/unstructured?style=social" width=90/></a> | <a href=https://pypi.org/project/unstructured><img src="https://img.shields.io/pypi/dw/unstructured" width=150/></a> |
| [Fivetran](https://www.fivetran.com/) | A platform that extracts, loads, and transforms data from various sources for analytics, AI, and operations | N/A | <a href=https://pypi.org/project/fivetran><img src="https://img.shields.io/pypi/dw/fivetran" width=150/></a> |
| [Airbyte](https://www.airbyte.com/) | An open-source data integration engine that helps consolidate data in data warehouses, lakes, and databases | <a href=https://github.com/airbytehq/airbyte><img src="https://img.shields.io/github/stars/airbytehq/airbyte?style=social" width=90/></a> | <a href=https://pypi.org/project/airbyte-cdk><img src="https://img.shields.io/pypi/dw/airbyte-cdk" width=150/></a> |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### Embedding Models
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [OpenAI Ada Embedding 2](https://platform.openai.com/docs/guides/embeddings) | OpenAI's most popular embedding model for capturing semantic relationships in text | n/a | <a href=https://pypi.org/project/openai><img src="https://img.shields.io/pypi/dw/openai" width=150/></a> |
| [Cohere AI](https://docs.cohere.com/docs/embeddings) | An independent commerical provider of LLMs, with particular focus on embeddings for semantic search, topic clustering, and vertical applications | <a href=https://github.com/cohere-ai/notebooks><img src="https://img.shields.io/github/stars/cohere-ai/notebooks?style=social" width=90/></a> | <a href=https://pypi.org/project/cohere><img src="https://img.shields.io/pypi/dw/cohere" width=150/></a> |
| [Sentence Transformers](https://www.sbert.net/) | An open-source Python framework for sentence, text, and image embeddings | <a href=https://github.com/UKPLab/sentence-transformers><img src="https://img.shields.io/github/stars/UKPLab/sentence-transformers?style=social" width=90/></a> | <a href=https://pypi.org/project/sentence-transformers><img src="https://img.shields.io/pypi/dw/sentence-transformers" width=150/></a> |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### Vector Databases
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [Pinecone](https://pinecone.io/) | A managed, cloud-native vector database with a simple API for high-performance AI applications | n/a | <a href=https://pypi.org/project/pinecone-client><img src="https://img.shields.io/pypi/dw/pinecone-client" width=150/></a> |
| [Weaviate](https://weaviate.io/) | An open-source vector database that stores both objects and vectors | <a href=https://github.com/weaviate/weaviate><img src="https://img.shields.io/github/stars/weaviate/weaviate?style=social" width=90/></a> | <a href=https://pypi.org/project/weaviate-client><img src="https://img.shields.io/pypi/dw/weaviate-client" width=150/></a> |
| [ChromaDB](https://trychroma.com/) | An AI-native, open-source embedding database platform for developers | <a href=https://github.com/chroma-core/chroma><img src="https://img.shields.io/github/stars/chroma-core/chroma?style=social" width=90/></a> | <a href=https://pypi.org/project/chromadb><img src="https://img.shields.io/pypi/dw/chromadb" width=150/></a> |
| [Pgvector](https://github.com/pgvector/pgvector) | An open-source vector similarity search for Postgres, allowing for exact and approximate nearest-neighbor search | <a href=https://github.com/pgvector/pgvector><img src="https://img.shields.io/github/stars/pgvector/pgvector?style=social" width=90/></a> | <a href=https://pypi.org/project/pgvector><img src="https://img.shields.io/pypi/dw/pgvector" width=150/></a> |
| [Zilliz (Milvus)](https://milvus.io/) | An open-source vector database, built for developing and maintaining AI applications | <a href=https://github.com/milvus-io/milvus><img src="https://img.shields.io/github/stars/milvus-io/milvus?style=social" width=90/></a> | <a href=https://pypi.org/project/pymilvus><img src="https://img.shields.io/pypi/dw/pymilvus" width=150/></a> |
| [Qdrant](https://qdrant.tech/) | A vector database and vector similarity search engine | <a href=https://github.com/qdrant/qdrant><img src="https://img.shields.io/github/stars/qdrant/qdrant?style=social" width=90/></a> | <a href=https://pypi.org/project/qdrant-client><img src="https://img.shields.io/pypi/dw/qdrant-client" width=150/></a> |
| [Metal io](https://getmetal.io/) | A managed service for developers to build applications with machine learning embeddings | N/A | <a href=https://pypi.org/project/metal-python><img src="https://img.shields.io/pypi/dw/metal-python" width=150/></a> |
| [LanceDB](https://lancedb.com/) | A serverless vector database for AI applications | <a href=https://github.com/lancedb/lancedb><img src="https://img.shields.io/github/stars/lancedb/lancedb?style=social" width=90/></a> | <a href=https://pypi.org/project/lancedb><img src="https://img.shields.io/pypi/dw/lancedb" width=150/></a> |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### Playgrounds
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [OpenAI Playground](https://platform.openai.com/) | A web-based platform for experimenting with various machine-learning models developed by OpenAI | N/A | N/A |
| [nat.dev](https://nat.dev) | A platform that allows users to test prompts with multiple language models and compare their performance | <a href=https://github.com/nat/openplayground><img src="https://img.shields.io/github/stars/nat/openplayground?style=social" width=90/></a> | <a href=https://pypi.org/project/openplayground><img src="https://img.shields.io/pypi/dw/openplayground" width=150/></a> |
| [Humanloop](https://humanloop.com/) | A platform that helps developers build applications on top of large language models | <a href=https://github.com/humanloop/humanloop-tutorial-python><img src="https://img.shields.io/github/stars/humanloop/humanloop-tutorial-python?style=social" width=90/></a> | <a href=https://pypi.org/project/humanloop><img src="https://img.shields.io/pypi/dw/humanloop" width=150/></a> |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### Orchestrators
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [Langchain](https://langchain.com/) | An open-source library that gives developers the tools to build applications powered by LLMs | <a href=https://github.com/langchain-ai/langchain><img src="https://img.shields.io/github/stars/langchain-ai/langchain?style=social" width=90/></a> | <a href=https://pypi.org/project/langchain><img src="https://img.shields.io/pypi/dw/langchain" width=150/></a> |
| [LlamaIndex](https://llamaindex.ai/) | A data framework for LLM applications to ingest, structure, and access private or domain-specific data | <a href=https://github.com/jerryjliu/llama_index><img src="https://img.shields.io/github/stars/jerryjliu/llama_index?style=social" width=90/></a> | <a href=https://pypi.org/project/llama-index><img src="https://img.shields.io/pypi/dw/llama-index" width=150/></a> |
| [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel) | A lightweight open-source orchestration SDK  | <a href=https://github.com/microsoft/semantic-kernel><img src="https://img.shields.io/github/stars/microsoft/semantic-kernel?style=social" width=90/></a> | <a href=https://pypi.org/project/semantic-kernel><img src="https://img.shields.io/pypi/dw/semantic-kernel" width=150/></a> |
| [Vercel AI SDK](https://sdk.vercel.ai/docs) | An open-source library for developers to build streaming UIs in JavaScript and TypeScript | <a href=https://github.com/vercel/ai><img src="https://img.shields.io/github/stars/vercel-labs/ai?style=social" width=90/></a> | <a href=https://pypi.org/project/vercel-ai-sdk><img src="https://img.shields.io/npm/dw/ai" width=150/></a>(node/npm)|
| [Vectara AI](https://vectara.com/) | A search and discovery platform for AI conversations utilizing your own data | <a href=https://github.com/vectara/vectara-ingest><img src="https://img.shields.io/github/stars/vectara/vectara-ingest?style=social" width=90/></a> | N/A |
| [ChatGPT](https://chat.openai.com) | An AI chatbot that uses natural language processing to create humanlike conversational dialogue | N/A| N/A |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### APIs / Plugins
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [Serp API](https://serpapi.com/) | A real-time API to access Google search results, as well as handling proxies, solving captchas, and parsing structured data | <a href=https://github.com/serpapi/google-search-results-python><img src="https://img.shields.io/github/stars/serpapi/google-search-results-python?style=social" width=90/></a> | <a href=https://pypi.org/project/google-search-results><img src="https://img.shields.io/pypi/dw/google-search-results" width=150/></a> |
| [Wolfram Alpha API](https://wolframalpha.com/) | A web-based API providing computational and presentation capabilities for integration into various applications | N/A | <a href=https://pypi.org/project/wolframalpha><img src="https://img.shields.io/pypi/dw/wolframalpha" width=150/></a> |
| [Zapier API AI Plugin](https://zapier.com/) | A plugin that allows you to connect 5,000+ apps and interact with them directly inside ChatGPT | N/A | N/A |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### LLM Caches
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [Redis](https://redis.io/) | An in-memory data structure store used as a database, cache, message broker, and streaming engine | <a href=https://github.com/redis/redis><img src="https://img.shields.io/github/stars/redis/redis?style=social" width=90/></a> | <a href=https://pypi.org/project/redis/><img src="https://img.shields.io/pypi/dw/redis" width=150/></a> |
| [SQLite](https://sqlite.org/) | A self-contained, serverless, zero-configuration, transactional SQL database engine | <a href=https://github.com/sqlite/sqlite><img src="https://img.shields.io/github/stars/sqlite/sqlite?style=social" width=90/></a> | <a href=https://pypi.org/project/pysqlite3/><img src="https://img.shields.io/pypi/dw/pysqlite3" width=150/></a> |
| [GPTCache](https://github.com/zilliztech/GPTCache) | An open-source tool for improving the efficiency and speed of GPT-based applications by implementing a cache to store the responses | <a href=https://github.com/zilliztech/GPTCache><img src="https://img.shields.io/github/stars/zilliztech/GPTCache?style=social" width=90/></a> | N/A |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### Logging / Monitoring / Eval
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [Arize AI](https://arize.com/) | An observability platform for both LLMs and supervised ML | <a href=https://github.com/Arize-ai><img src="https://img.shields.io/github/stars/Arize-ai?style=social" width=90/></a> | <a href=https://pypi.org/project/arize><img src="https://img.shields.io/pypi/dw/arize" width=150/></a> |
| [Weights & Biases](https://wandb.ai) | An MLOps platform for streamlining ML workflows | <a href=https://github.com/wandb/wandb><img src="https://img.shields.io/github/stars/wandb/wandb?style=social" width=90/></a> | <a href=https://pypi.org/project/wandb><img src="https://img.shields.io/pypi/dw/wandb" width=150/></a> |
| [MLflow](https://mlflow.org/docs/latest/llm-tracking.html) | A platform to streamline machine learning development | <a href=https://github.com/mlflow/mlflow><img src="https://img.shields.io/github/stars/mlflow/mlflow?style=social" width=90/></a> | <a href=https://pypi.org/project/mlflow><img src="https://img.shields.io/pypi/dw/mlflow" width=150/></a> |
| [PromptLayer](https://promptlayer.com/) | A platform for tracking, managing, and sharing LLM prompt engineering | <a href=https://github.com/MagnivOrg/prompt-layer-library><img src="https://img.shields.io/github/stars/MagnivOrg/prompt-layer-library?style=social" width=90/></a> | <a href=https://pypi.org/project/promptlayer><img src="https://img.shields.io/pypi/dw/promptlayer" width=150/></a> |
| [Helicone](https://helicone.ai/) | An open-source observability platform for LLMs | <a href=https://github.com/Helicone/helicone><img src="https://img.shields.io/github/stars/Helicone/helicone?style=social" width=90/></a> | <a href=https://pypi.org/project/helicone><img src="https://img.shields.io/pypi/dw/helicone" width=150/></a> |
| [Portkey AI](https://portkey.ai/) | Platform to develop, launch, maintain, and iterate generative AI apps and features  | N/A | N/A |
| [Freeplay AI](https://freeplay.ai/) | Platform to prototype, test, and optimize LLM features for customers | N/A | N/A |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>


### Validators
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [Guardrails AI](https://shreyar.github.io/guardrails/) | An open-source Python package for specifying structure and type, validating and correcting the outputs of LLMs | <a href=https://github.com/ShreyaR/guardrails><img src="https://img.shields.io/github/stars/ShreyaR/guardrails?style=social" width=90/></a> | <a href=https://pypi.org/project/guardrails-ai><img src="https://img.shields.io/pypi/dw/guardrails-ai" width=150/></a> |
| [Rebuff](https://github.com/woop/rebuff) | An open-source framework designed to detect and protect against prompt injection attacks in LLM apps | <a href=https://github.com/woop/rebuff><img src="https://img.shields.io/github/stars/woop/rebuff?style=social" width=90/></a> | <a href=https://pypi.org/project/rebuff><img src="https://img.shields.io/pypi/dw/rebuff" width=150/></a> |
| [Microsoft Guidance](https://github.com/microsoft/guidance) | A guidance language for controlling LLMs, providing a syntax for architecting LLM workflows | <a href=https://github.com/microsoft/guidance><img src="https://img.shields.io/github/stars/microsoft/guidance?style=social" width=90/></a> | <a href=https://pypi.org/project/guidance><img src="https://img.shields.io/pypi/dw/guidance" width=150/></a> |
| [LMQL](https://lmql.ai/) | An open-source programming language and platform for language model interaction | <a href=https://github.com/eth-sri/lmql><img src="https://img.shields.io/github/stars/eth-sri/lmql?style=social" width=90/></a> | <a href=https://pypi.org/project/lmql><img src="https://img.shields.io/pypi/dw/lmql" width=150/></a> |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### LLM APIs (proprietary)
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [OpenAI](https://openai.com) | Provide many leading LLMs, including the GPT-3.5 and GPT-4 families. | N/A | <a href=https://pypi.org/project/openai><img src="https://img.shields.io/pypi/dw/openai" width=150/></a> |
| [Anthropic](https://anthropic.com) | Developer of Claude, an AI assistant based on Anthropic’s research  | N/A | <a href=https://pypi.org/project/anthropic><img src="https://img.shields.io/pypi/dw/anthropic" width=150/></a> |
| [Cohere AI](https://docs.cohere.com/docs/embeddings) | LLM vendor with particular focus on embeddings for semantic search, topic clustering, and vertical applications | <a href=https://github.com/cohere-ai/notebooks><img src="https://img.shields.io/github/stars/cohere-ai/notebooks?style=social" width=90/></a> | <a href=https://pypi.org/project/cohere><img src="https://img.shields.io/pypi/dw/cohere" width=150/></a> |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### LLM APIs (open source)
| Name (site) | Description | Github | Pip Installs |
|------------|------------|------------|------------|
| [Hugging Face](https://huggingface.co/) | A hub for open-source AI models and inference endpoints, including leading base LLMs and LoRAs/fine-tunes | <a href=https://github.com/huggingface/transformers><img src="https://img.shields.io/github/stars/huggingface/transformers?style=social" width=90/></a> | <a href=https://pypi.org/project/transformers><img src="https://img.shields.io/pypi/dw/transformers" width=150/></a> |
| [Replicate](https://replicate.com/) | An AI hosting platform and model inference hub that allows software developers to integrate AI models into their apps | <a href=https://github.com/replicate/cog><img src="https://img.shields.io/github/stars/replicate/cog?style=social" width=90/></a> | <a href=https://pypi.org/project/replicate><img src="https://img.shields.io/pypi/dw/replicate" width=150/></a> |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### App Hosting Platforms
| Name (site) | Description |
|------------|------------|
| [Vercel](https://vercel.com/) | A cloud platform designed for front-end engineers, built with first-class support for LLM apps. |
| [Netlify](https://www.netlify.com/) | An enterprise cloud computing company that offers a development platform for web applications and dynamic websites. |
| [Steamship](https://steamship.com/) | An SDK and hosting platform for AI agents and tools, both a package manager and package hosting service for AI. |
| [Streamlit](https://streamlit.io/) | An open-source Python library designed for creating and sharing custom web apps for machine learning and data science. |
| [Modal](https://modal.com) | A platform that enables running distributed applications using the modal Python package. | 

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### Cloud Providers
| Name (site) | Description |
|------------|------------|
| [Amazon Web Services](https://aws.amazon.com/) | A cloud computing platform, offering services from data centers globally. |
| [Google Cloud Platform](https://cloud.google.com/) | A cloud computing platform, offering services from data centers globally. |
| [Microsoft Azure](https://azure.microsoft.com/) | A cloud computing platform, offering services from data centers globally. |
| [CoreWeave](https://coreweave.com/) | A specialized cloud provider that delivers GPUs on top of flexible deployment infrastructure. |

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

### Opinionated Clouds
| Name (site) | Description |
|------------|------------|
| [Databricks (MosaicML)](https://databricks.com/) | Databricks acquired Mosaic ML in 2023, along with its tooling and platform for efficient pre-trainining, fine-tuning and inferencing LLMs. |  
| [Anyscale](https://anyscale.com/) | A fully managed compute platform that enables the development, deployment, and management of Ray (Python) apps. |
| [Modal](https://modal.com) | A platform that eables running distributed applications using the Modal Python package. | 
| [Runpod](https://runpod.io/) | A cloud computing platform designed for AI and machine learning applications. |
| [OctoML](https://octoml.ai/) | A compute service that allows users to run, tune, and scale generative models. |


<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>

<br>
<br>
<br>


## Formatting Prompt Templates
We were able to partly automate this - particularly finding Github and PyPI links - using this [Perplexity search prompt](https://github.com/a16z-infra/llm-app-stack/blob/main/table_construction_prompts/prompt_1_search.txt). It worked roughly ~75% of the time and could handle ~3 projects at a time, pulling data from 20-30 sources in each iteration. 

This [Chatgpt prompt](https://github.com/a16z-infra/llm-app-stack/blob/main/table_construction_prompts/prompt_2_format.txt) is also helpful for formatting markdown. It works best with gpt4, but can be used with gpt3 as well.

<p style="text-align: right;"><a href="#table-of-contents">^ Back to Contents ^</a></p>
