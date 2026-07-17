# Requisitos de Máquina — RAG Institucional UFPel

Levantamento de requisitos de hardware e software para hospedar a aplicação RAG
(Streamlit + LangChain + PostgreSQL/pgvector), com base na arquitetura atual do
projeto e em medições do ambiente de desenvolvimento (julho/2026).

---

## 1. Arquitetura e o que roda na máquina

| Componente | Onde executa | Impacto no dimensionamento |
|---|---|---|
| LLM de geração (deepseek-v4-pro/flash) | **API externa** (NVIDIA NIM / OpenRouter) | Nenhum — só rede |
| Embeddings (nv-embedqa-e5-v5, 1024 dims) | **API externa** (NVIDIA NIM) | Nenhum — só rede |
| Reranker primário (NVIDIA rerank) | **API externa** | Nenhum |
| Reranker fallback (cross-encoder ms-marco-MiniLM-L-6-v2) | **Local, CPU** (PyTorch) | ~22 MB de modelo + ~1 GB de runtime PyTorch |
| Banco vetorial (PostgreSQL 16 + pgvector, índices HNSW + GIN trigram) | **Local** | Hoje: 2,7 GB de banco, 36.492 vetores |
| Busca híbrida BM25 (rank-bm25) | **Local, em memória** | Corpus dos chunks carregado em RAM |
| Interface (Streamlit) + pipeline LangChain | **Local** | ~1,5–2,5 GB RSS por processo |
| Avaliação (BERTScore/ROUGE) | **Local, CPU** — uso pontual | Modelo ~1,4 GB carregado só durante avaliação |
| Crawler + ingestão (aiohttp, lotes de 500 chunks) | **Local** — uso pontual | I/O de rede + CPU moderada |
| pgAdmin 4 (opcional) | Local | ~300 MB RAM |

**Ponto-chave:** toda a inferência pesada (LLM, embeddings, rerank primário) é
via API. **A máquina não precisa de GPU.** Os fallbacks locais rodam em CPU.

## 2. Medições do ambiente atual (base do dimensionamento)

- Banco `semanticdb`: **2,7 GB** com **36.492 embeddings** de 1024 dims (HNSW + GIN)
- Ambiente Python (`.venv` com torch, sentence-transformers, langchain): **5,8 GB**
- Cache de modelos HuggingFace (`~/.cache/huggingface`): **até 4,4 GB**
- Dados crawleados (`dados_ufpel.json`): ~6 MB por execução do crawler
- Máquina de desenvolvimento onde tudo roda confortavelmente: 4 cores / 16 GB RAM

## 3. Requisitos mínimos (uso didático / demonstração, 1–3 usuários)

| Recurso | Mínimo | Justificativa |
|---|---|---|
| **CPU** | 2 vCPUs (x86-64) | PostgreSQL + Streamlit + cross-encoder em CPU; consultas HNSW são leves nesse volume |
| **RAM** | 8 GB | ~2 GB app Python (PyTorch/LangChain/BM25) + ~2 GB PostgreSQL + folga para SO. Com 4 GB há risco de OOM quando o fallback local de rerank ou o BERTScore carregam |
| **Disco** | 30 GB **SSD** | SO (~10 GB) + venv (6 GB) + cache de modelos (até 4,4 GB) + banco (2,7 GB, cresce com ingestão) + logs. HNSW exige disco rápido para boa latência |
| **GPU** | Não requerida | Inferência via API |
| **Rede** | 10 Mbps estáveis, saída HTTPS liberada | Acesso a `integrate.api.nvidia.com` e `openrouter.ai`; crawler acessa portal da UFPel |
| **SO / Software** | Ubuntu 24.04 LTS, Python 3.12, PostgreSQL 16 + pgvector 0.6+, `build-essential`, `libpq-dev` | Versões testadas pelo `setup_ambiente.sh` |

Com essa configuração a aplicação funciona, mas ingestão completa + consultas
simultâneas + avaliação BERTScore não devem rodar ao mesmo tempo.

## 4. Requisitos recomendados (produção leve / uso contínuo, ~5–20 usuários simultâneos)

| Recurso | Recomendado | Justificativa |
|---|---|---|
| **CPU** | 4–8 vCPUs | Paralelismo entre sessões Streamlit, consultas Postgres (HNSW + GIN trigram) e rerank local eventual; ingestão/crawler sem degradar o serviço |
| **RAM** | 16 GB | ~4 GB PostgreSQL (`shared_buffers` 4 GB mantém índices HNSW quentes) + 3–4 GB app + BM25 em memória + avaliação + margem de crescimento do corpus |
| **Disco** | 60–100 GB **NVMe SSD** | Crescimento do banco com novas ingestões (regra prática: ~1 GB a cada ~13 mil chunks, índices inclusos), backups locais e logs (`app.log`, `reliability.jsonl`) |
| **GPU** | Não requerida | — |
| **Rede** | 50+ Mbps, latência < 100 ms para as APIs | Latência de rede domina o tempo de resposta (embeddings + LLM por consulta) |
| **Backup** | `pg_dump` diário + snapshot do disco | O banco é o único estado difícil de reconstruir (reingestão custa horas de API) |

## 5. Ajustes recomendados no PostgreSQL (máquina de 16 GB)

```conf
shared_buffers = 4GB            # mantém HNSW/GIN em cache
effective_cache_size = 10GB
work_mem = 64MB
maintenance_work_mem = 1GB      # acelera (re)construção de índices HNSW na ingestão
max_connections = 50
```

## 6. Regras de escala (se o corpus crescer)

- **Vetores:** cada 100 mil chunks de 1024 dims ≈ +0,4 GB de dados brutos e
  +0,5–1 GB de índice HNSW. Some isso à RAM recomendada para manter o índice em cache.
- **Usuários:** cada sessão Streamlit ativa custa ~50–150 MB; acima de ~30
  simultâneos, considere separar PostgreSQL em outra máquina.
- **Sem chave NVIDIA/OpenRouter a aplicação não funciona** — embeddings são
  obrigatórios via NVIDIA NIM (`get_embeddings()` falha sem `NVIDIA_API_KEY`).

## 7. Resumo

| | Mínimo | Recomendado |
|---|---|---|
| CPU | 2 vCPUs | 4–8 vCPUs |
| RAM | 8 GB | 16 GB |
| Disco | 30 GB SSD | 60–100 GB NVMe SSD |
| GPU | — | — |
| Rede | 10 Mbps, HTTPS liberado | 50+ Mbps, latência < 100 ms |
| SO | Ubuntu 24.04 LTS / Python 3.12 / PostgreSQL 16 + pgvector | idem |
