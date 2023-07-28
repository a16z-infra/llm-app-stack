# LLM App Stack

*aka Emerging Architectures for LLM Applications*

![llm-app-stack](https://github.com/a16z-infra/llm-app-stack/assets/26883865/92734642-9651-4aaa-a803-79a6cf5414ef)

This is a list of available tools, projects, and vendors at each layer of the LLM app stack. Our [original article](https://a16z.com/2023/06/20/emerging-architectures-for-llm-applications/) included only the most popular tools based on user interviews. This repo attempts to be comprehensive, covering all options in each category. If you see anything missing from list, or miscategorized, please open a PR!

### Data Pipelines
| Name | Description | Website | Github | Pip Installs | Npm Installs |
| --- | --- | --- | --- | --- | --- |
| Databricks | A unified set of tools for building, deploying, sharing, and maintaining enterprise-grade data solutions at scale | [databricks.com](https://databricks.com/) | ![GitHub Repo stars](https://img.shields.io/github/stars/apache/spark?style=social) | <a href=https://pypi.org/project/pyspark><img src="https://img.shields.io/pypi/dw/pyspark" width=150/></a> | <a href=https://www.npmjs.com/package/@databricks/sql><img src="https://img.shields.io/npm/dw/@databricks/sql" width=150/></a>   |
| Airflow | A platform to programmatically author, schedule, and monitor workflows | [airflow.apache.org](https://airflow.apache.org/) | ![GitHub Repo stars](https://img.shields.io/github/stars/apache/airflow?style=social) | <a href=https://pypi.org/project/apache-airflow><img src="https://img.shields.io/pypi/dw/apache-airflow" width=150/></a> | <a href=https://www.npmjs.com/package/@backstage/plugin-apache-airflow><img src="https://img.shields.io/npm/dw/@backstage/plugin-apache-airflow" width=150/></a>   |
| Unstructured.io | Open-source components for pre-processing text documents such as PDFs, HTML and Word Documents | [unstructured.io](https://unstructured.io/) | ![Unstructured-IO/unstructured](https://img.shields.io/github/stars/Unstructured-IO/unstructured?style=social) | <a href=https://pypi.org/project/unstructured><img src="https://img.shields.io/pypi/dw/unstructured" width=150/></a> | n/a |

| Name (site) | Description | Github | Pip Installs | Npm Installs |
| --- | --- | --- | --- | --- |
| [Databricks](https://databricks.com/) | A unified set of tools for building, deploying, sharing, and maintaining enterprise-grade data solutions at scale | ![GitHub Repo stars](https://img.shields.io/github/stars/apache/spark?style=social) | <a href=https://pypi.org/project/pyspark><img src="https://img.shields.io/pypi/dw/pyspark" width=150/></a> | <a href=https://www.npmjs.com/package/@databricks/sql><img src="https://img.shields.io/npm/dw/@databricks/sql" width=150/></a> |
| [Airflow](https://airflow.apache.org/) | A platform to programmatically author, schedule, and monitor workflows | ![GitHub Repo stars](https://img.shields.io/github/stars/apache/airflow?style=social) | <a href=https://pypi.org/project/apache-airflow><img src="https://img.shields.io/pypi/dw/apache-airflow" width=150/></a> | <a href=https://www.npmjs.com/package/@backstage/plugin-apache-airflow><img src="https://img.shields.io/npm/dw/@backstage/plugin-apache-airflow" width=150/></a> |
| [Unstructured.io](https://unstructured.io/) | Open-source components for pre-processing text documents such as PDFs, HTML and Word Documents | ![Unstructured-IO/unstructured](https://img.shields.io/github/stars/Unstructured-IO/unstructured?style=social) | <a href=https://pypi.org/project/unstructured><img src="https://img.shields.io/pypi/dw/unstructured" width=150/></a> | n/a |


### Embedding Model
| Name | Description | Website | Github | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Open AI | The OpenAI API provides a general-purpose "text in, text out" interface to AI models, suitable for any English language task. It can generate text completions, be integrated into products, and is trainable with datasets or human feedback.| [openai/openai](https://github.com/openai/openai) [8] | openai [16] | [openai](https://www.npmjs.com/package/openai) [15] |
| Cohere AI | AI models that power interactive chat features, generate text for product descriptions, blog posts, and articles, and capture the meaning of text for search, content moderation, and intent recognition [10] | [cohere.com](https://cohere.com/) | | cohere [12] | [cohere-ai](https://www.npmjs.com/package/cohere-ai) [7] |
| HuggingFace | Provides open-source components for pre-processing text documents such as PDFs, HTML and Word Documents [2] | [huggingface.co](https://huggingface.co/) | | | ![npm](https://img.shields.io/npm/dw/transformers) |


### Vector Database
| Name | Description | Website | Github | Pip Installs | Npm Installs |
| --- | --- | --- | --- | --- | --- |
| Pinecone | The vector database for machine learning applications. Build vector-based personalization, ranking, and search systems that are accurate, fast, and scalable. | [pinecone.io](https://pinecone.io/) | [github.com/pinecone-io](https://github.com/pinecone-io) | pinecone-client | |
| Weaviate | An open source vector database that stores both objects and vectors. This allows for combining vector search with structured filtering. | [weaviate.io](https://weaviate.io/) | [github.com/weaviate/weaviate](https://github.com/weaviate/weaviate) | weaviate-client | weaviate-client |
| ChromaDB | Developer of an open-source embedding database platform intended to help developers add state and memory to their artificial intelligence-enabled applications. | [trychroma.com](https://trychroma.com/) | | | |
| pgvector | Open-source vector similarity search for Postgres. | [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) | [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) | | |

### Data Pipelines
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Databricks | | [databricks.com](https://databricks.com/) | | | |
| Airflow | | [airflow.apache.org](https://airflow.apache.org/) | | | |
| Unstructured.io | | [unstructured.io](https://unstructured.io/) | | | |

### Embedding Model
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Open AI | | [openai.com](https://openai.com/) | | | |
| Cohere AI | | [cohere.com](https://cohere.com/) | | | |
| HuggingFace | | [huggingface.co](https://huggingface.co/) | | | |

### Vector Database
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Pinecone | | [pinecone.io](https://pinecone.io/) | | | |
| Weaviate | | [weaviate.io](https://weaviate.io/) | | | |
| ChromaDB | | [trychroma.com](https://trychroma.com/) | | | |
| pgvector | | [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) | | | |



### Playground
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| OpenAI | | [openai.com](https://openai.com/) | | | |
| nat.dev | | [nat.dev](https://nat.dev) | | | |
| Humanloop | | [humanloop.com](https://humanloop.com/) | | | |

### Orchestration
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Langchain | | [langchain.com](https://langchain.com/) | | | |
| LlamaIndex | | [llamaindex.ai](https://llamaindex.ai/) | | | |
| ChatGPT | | [chat.openai.com](https://chat.openai.com) | | | |

### APIs / Plugins
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Serp | | [serpapi.com](https://serpapi.com/) | | | |
| Wolfram | | [wolframalpha.com](https://wolframalpha.com/) | | | |
| Zapier | | [zapier.com](https://zapier.com/) | | | |

### LLM Cache
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Redis | | [redis.io](https://redis.io/) | | | |
| SQLite | | [sqlite.org](https://sqlite.org/) | | | |
| GPTCache | | [github.com/zilliztech/GPTCache](https://github.com/zilliztech/GPTCache) | | | |

### Logging / LLMops
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Weights & Biases | | [wandb.ai](https://wandb.ai) | | | |
| MLflow | | [mlflow.org/docs/latest/llm-tracking.html](https://mlflow.org/docs/latest/llm-tracking.html) | | | |
| PromptLayer | | [promptlayer.com](https://promptlayer.com/) | | | |
| Helicone | | [helicone.ai](https://helicone.ai/) | | | |

### Validation
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Guardrails | | [shreyar.github.io/guardrails/](https://shreyar.github.io/guardrails/) | | | |
| Rebuff | | [github.com/woop/rebuff](https://github.com/woop/rebuff) | | | |
| Microsoft Guidance | | [github.com/microsoft/guidance](https://github.com/microsoft/guidance) | | | |
| LMQL | | [lmql.ai](https://lmql.ai/) | | | |

### App Hosting
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Vercel | | [vercel.com](https://vercel.com/) | | | |
| Steamship | | [steamship.com](https://steamship.com/) | | | |
| Streamlit | | [streamlit.io](https://streamlit.io/) | | | |
| Modal | | [modal.com](https://modal.com) | | | |

### LLM APIs (proprietary)
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| OpenAI | | [openai.com](https://openai.com/) | | | |
| Anthropic | | [anthropic.com](https://anthropic.com/) | | | |

### LLM APIs (open source)
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| HuggingFace | | [huggingface.co](https://huggingface.co/) | | | |
| Replicate | | [replicate.com](https://replicate.com/) | | | |

### Cloud Providers
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| AWS | | [aws.amazon.com](https://aws.amazon.com/) | | | |
| GCP | | [cloud.google.com](https://cloud.google.com/) | | | |
| Azure | | [azure.microsoft.com/en-us](https://azure.microsoft.com/en-us) | | | |
| Coreweave | | [coreweave.com](https://coreweave.com/) | | | |

### Opinionated Clouds
| Name | Description | Website | Github (stars) | Pip Installs | Npm Downloads |
| --- | --- | --- | --- | --- | --- |
| Databricks | | [databricks.com](https://databricks.com/) | | | |
| Anyscale | | [anyscale.com](https://anyscale.com/) | | | |
| Mosaic | | [mosaicml.com](https://mosaicml.com) | | | |
| Modal | | [modal.com](https://modal.com) | | | |
| Runpod | | [runpod.io](https://runpod.io/) | | | |


Criteria for inclusion and contribution guide
