# Architecture Deep Dive

## System Design
The system follows a **Modular Translation Pattern**. Instead of training a monolithic model to "guess" the answer, we use a multi-stage pipeline:

### 1. Schema-Driven RAG
To handle large-scale enterprise schemas, we don't dump every table into the prompt. 
- **Component:** `SchemaRetriever` (ChromaDB).
- **Process:** The Natural Language (NL) query is embedded, and the top-2 most relevant table schemas are retrieved and injected into the LLM context.

### 2. The Hybrid Synthetic Generation (Skeleton-to-Sample)
To avoid "lazy" synthetic data, we used a two-step process:
- **Step A (Python):** Generates a logical "Skeleton" (e.g., `SOURCE {{TABLE}} |> FILTER {{COL}} == {{VAL}}`).
- **Step B (Qwen-235B):** Performs "Semantic Infilling." It replaces placeholders with realistic values and creates 3 linguistic variations (Casual, Formal, Indirect).

### 3. Fine-tuning Strategy
We used **Phi-4** due to its superior reasoning capabilities. 
- **LoRA Configuration:** `r=64, alpha=128`. 
- **Target Modules:** We targeted all linear layers + `embed_tokens` and `lm_head`. This is essential for DSL tasks where the model must learn a new vocabulary distribution and strict punctuation rules.

### 4. Safe Transpilation Layer
To bridge the gap between AI and Data, we built the `AuraTranspiler`. It converts AuraDSL to Parameterized SQL, preventing SQL injections and providing a bridge for execution-based validation.