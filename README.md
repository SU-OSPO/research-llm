# Research LLM System

An integrated research assistant built with retrieval, structured data pipelines, and OpenAlex ingestion.

---

## Architecture Overview

```
research-llm/
├── rag/        # Retrieval-Augmented Generation system
├── database/   # Data ingestion, processing, and storage
├── openalex/   # OpenAlex data pipeline
```

---

## Modules

### RAG System

Core retrieval and question answering engine.

* Semantic search
* Graph-based retrieval
* Streamlit interface
* Session memory

📂 Location: `/rag`
📖 Details: [RAG README](./rag/README.md)

---

### Database Pipeline

Handles ingestion, preprocessing, and enrichment of research data.

* PDF ingestion
* Data cleaning and normalization
* Model training utilities
* Pipeline orchestration

📂 Location: `/database`
📖 Details: [Database README](./database/README.md)

---

### OpenAlex Integration

Fetches and processes academic metadata from OpenAlex.

* Paper metadata retrieval
* Download pipelines
* Chunking and normalization
* Graph + vector ingestion

📂 Location: `/openalex`
📖 Details: [OpenAlex README](./openalex/README.md)

---

## Workflow

1. OpenAlex → fetch research papers
2. Database → clean, structure, enrich
3. RAG → retrieve + answer queries

---

## Quick Start

```bash
# Clone repo
git clone https://github.com/AryanApte1408/research-llm.git
cd research-llm
```

Navigate to a module:

```bash
cd rag
# or
cd database
# or
cd openalex
```

Follow instructions in each module’s README.

---

## Tech Stack

* Python
* ChromaDB
* Neo4j
* Streamlit
* OpenAlex API

---

## Notes

Each module is independently runnable but designed to work together as a pipeline.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.

## Support

Developed and maintained by the [Open Source Program Office](https://opensource.syracuse.edu/) at Syracuse University. Reach out for feedback and suggested improvements:

- [GitHub Issues](../../issues)
- [Email](mailto:ospo@syr.edu)

## Acknowledgments

This project was supported as part of grants (#[G2023-20946](https://sloan.org/grant-detail/g-2023-20946), #[G-2025-79206](https://sloan.org/grant-detail/g-2025-79206)) from the Alfred P. Sloan Foundation.
