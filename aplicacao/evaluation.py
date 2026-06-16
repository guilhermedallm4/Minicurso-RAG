"""
Métricas de Avaliação do Pipeline RAG
=============================================================================
Conceito — Módulo Avançado

Avaliar um sistema RAG exige métricas para dois componentes distintos:

  ┌─────────────────────────────────────────────────────────────┐
  │  RETRIEVAL                    GENERATION                    │
  │  ─────────────────────        ────────────────────────────  │
  │  MRR@K  Mean Reciprocal       BERTScore  semântica          │
  │         Rank                             token-a-token      │
  │  R@K    Recall at K           ROUGE-L    sobreposição       │
  │  P@K    Precision at K                   de sequências      │
  │                               LLM Judge  avaliação          │
  │                                          estruturada        │
  └─────────────────────────────────────────────────────────────┘

MRR@K (Mean Reciprocal Rank):
  Para cada query, qual a posição do primeiro documento relevante?
  MRR = (1/|Q|) × Σ 1/rank_i
  MRR=1.0 → sempre retorna o relevante em 1º lugar
  MRR=0.5 → relevante está em média na 2ª posição

BERTScore:
  Computa a similaridade semântica token-a-token entre resposta gerada
  e referência usando embeddings de um modelo BERT pré-treinado.
  Métricas: Precisão, Recall e F1 no espaço de embeddings.

  Opção Lite: usa a API de embeddings (NVIDIA/Gemini) em vez de
  um modelo BERT local — mais leve, aproxi-mação razoável.

ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation):
  Mede a maior subsequência comum entre geração e referência.
  Puramente léxico, sem downloads, zero latência.

LLM Judge:
  Usa o próprio LLM para avaliar fidelidade, relevância e completude.
  Ideal quando não há resposta de referência disponível.
  Retorna scores 0–10 para cada dimensão.
"""
import numpy as np
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


# =============================================================================
# RETRIEVAL: MRR, Recall@K, Precision@K
# =============================================================================

def mrr_at_k(
    queries: List[Dict[str, Any]],
    retrieval_fn,
    k: int = 5,
) -> Dict[str, float]:
    """
    Calcula MRR@K, Recall@K e Precision@K sobre um conjunto de teste.

    Args:
        queries      : lista de dicts com chaves "query" e "relevant_text"
                       (substring ou texto exato do documento relevante)
        retrieval_fn : função que recebe query e retorna List[Document]
        k            : número de documentos a recuperar

    Returns:
        dict com mrr, recall_at_k, precision_at_k, e detalhes por query
    """
    reciprocal_ranks = []
    recalls = []
    precisions = []
    details = []

    for item in queries:
        query    = item["query"]
        relevant = item["relevant_text"].lower()

        docs = retrieval_fn(query)[:k]
        texts = [d.page_content.lower() for d in docs]

        # Encontra o rank do primeiro documento relevante
        rr = 0.0
        hits = 0
        for rank, text in enumerate(texts, 1):
            if relevant in text or text in relevant:
                if rr == 0.0:
                    rr = 1.0 / rank
                hits += 1

        recall    = 1.0 if hits > 0 else 0.0  # 1 doc relevante esperado
        precision = hits / len(texts) if texts else 0.0

        reciprocal_ranks.append(rr)
        recalls.append(recall)
        precisions.append(precision)
        details.append({"query": query, "rr": rr, "recall": recall, "rank": int(1/rr) if rr > 0 else -1})

    return {
        "mrr":          round(float(np.mean(reciprocal_ranks)), 4),
        "recall_at_k":  round(float(np.mean(recalls)), 4),
        "precision_at_k": round(float(np.mean(precisions)), 4),
        "k":            k,
        "n_queries":    len(queries),
        "details":      details,
    }


# =============================================================================
# GENERATION: BERTScore (Lite e Full), ROUGE-L, LLM Judge
# =============================================================================

def bertscore_lite(
    predictions: List[str],
    references: List[str],
) -> Dict[str, Any]:
    """
    BERTScore aproximado usando a API de embeddings configurada.
    Não requer nenhum modelo local — usa NVIDIA NIM ou Gemini.

    Calcula similaridade cosseno entre os embeddings da resposta
    gerada e da resposta de referência.

    Limitação: opera no nível de sentença (não token-a-token como o
    BERTScore original), mas é suficiente para avaliação comparativa.
    """
    from providers import get_embeddings

    emb_model = get_embeddings()
    scores = []

    for pred, ref in zip(predictions, references):
        p_vec = np.array(emb_model.embed_query(pred))
        r_vec = np.array(emb_model.embed_query(ref))
        cos_sim = float(np.dot(p_vec, r_vec) / (np.linalg.norm(p_vec) * np.linalg.norm(r_vec)))
        scores.append(cos_sim)

    return {
        "method":   "bertscore_lite (embedding cosseno)",
        "f1_mean":  round(float(np.mean(scores)), 4),
        "f1_min":   round(float(np.min(scores)), 4),
        "f1_max":   round(float(np.max(scores)), 4),
        "scores":   [round(s, 4) for s in scores],
    }


def bertscore_full(
    predictions: List[str],
    references: List[str],
    lang: str = "pt",
    model_type: str = "distilbert-base-multilingual-cased",
) -> Dict[str, Any]:
    """
    BERTScore completo (token-a-token) via pacote 'bert_score'.
    Faz download do modelo na primeira execução (~270 MB).

    Args:
        lang       : idioma do texto ('pt' para português)
        model_type : modelo BERT a usar (multilingual para PT-BR)
    """
    try:
        from bert_score import score as _bert_score
    except ImportError:
        return {"error": "Instale: pip install bert-score"}

    print(f"[BERTScore] Usando modelo '{model_type}'...")
    P, R, F1 = _bert_score(
        predictions, references,
        lang=lang,
        model_type=model_type,
        verbose=False,
    )
    return {
        "method":       f"bertscore_full ({model_type})",
        "precision":    round(float(P.mean()), 4),
        "recall":       round(float(R.mean()), 4),
        "f1_mean":      round(float(F1.mean()), 4),
        "f1_scores":    [round(f, 4) for f in F1.tolist()],
    }


def rouge_l_score(prediction: str, reference: str) -> Dict[str, float]:
    """
    ROUGE-L: maior subsequência comum normalizada.
    Puramente léxico, zero dependências extras além de rouge_score.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        result = scorer.score(reference, prediction)
        rl = result["rougeL"]
        return {
            "precision": round(rl.precision, 4),
            "recall":    round(rl.recall, 4),
            "f1":        round(rl.fmeasure, 4),
        }
    except ImportError:
        return {"error": "Instale: pip install rouge-score"}


def llm_judge(
    query: str,
    response: str,
    context: str,
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """
    LLM-as-a-Judge: avalia a resposta gerada em três dimensões (0–10).

    Dimensões avaliadas:
      fidelidade   : a resposta está fundamentada no contexto? (sem alucinação)
      relevância   : a resposta responde à pergunta do usuário?
      completude   : a resposta cobre todos os aspectos da pergunta?
    """
    from providers import get_llm
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import json, re

    referencia_str = f"\nResposta de referência:\n{reference}" if reference else ""

    prompt = ChatPromptTemplate.from_template(
        "Avalie a resposta abaixo em três dimensões usando notas de 0 a 10.\n"
        "Retorne EXCLUSIVAMENTE um JSON válido, sem explicações.\n\n"
        "Pergunta: {query}\n\n"
        "Contexto recuperado:\n{context}\n\n"
        "Resposta gerada:\n{response}"
        "{referencia}\n\n"
        "Retorne apenas:\n"
        '{{ "fidelidade": <0-10>, "relevancia": <0-10>, "completude": <0-10>, "justificativa": "<frase>" }}'
    )
    chain = prompt | get_llm() | StrOutputParser()
    raw = chain.invoke({
        "query": query,
        "context": context[:2000],
        "response": response,
        "referencia": referencia_str,
    })

    try:
        # Extrai o JSON mesmo que o LLM adicione texto ao redor
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group()) if match else {}
        result["score_medio"] = round(
            (result.get("fidelidade", 0) + result.get("relevancia", 0) +
             result.get("completude", 0)) / 3, 2
        )
        return result
    except Exception:
        return {"raw": raw, "error": "JSON inválido retornado pelo LLM"}


# =============================================================================
# SUITE COMPLETA
# =============================================================================

def evaluate_pipeline(
    test_set: List[Dict[str, Any]],
    retrieval_fn,
    generation_fn,
    k: int = 5,
    use_bertscore_full: bool = False,
) -> Dict[str, Any]:
    """
    Executa a suite de avaliação completa sobre um conjunto de testes.

    Args:
        test_set: lista de dicts com:
            - "query"         : pergunta
            - "relevant_text" : trecho do documento correto (para MRR)
            - "reference"     : resposta esperada (para BERTScore/ROUGE)
        retrieval_fn  : fn(query) → List[Document]
        generation_fn : fn(query) → str
        k             : número de docs para MRR
        use_bertscore_full: se True usa bert_score (precisa do modelo local)
    """
    print(f"\n[Avaliação] Iniciando para {len(test_set)} queries...\n")

    # Retrieval metrics
    retrieval_metrics = mrr_at_k(test_set, retrieval_fn, k=k)

    predictions = [generation_fn(item["query"]) for item in test_set]
    references  = [item.get("reference", "") for item in test_set]

    # Generation metrics
    bs_lite  = bertscore_lite(predictions, references) if any(references) else {}
    bs_full  = bertscore_full(predictions, references) if use_bertscore_full else {}
    rouge    = {
        "rouge_l": [rouge_l_score(p, r) for p, r in zip(predictions, references)]
    }
    rouge["rouge_l_mean_f1"] = round(
        float(np.mean([s["f1"] for s in rouge["rouge_l"] if "f1" in s])), 4
    ) if rouge["rouge_l"] else 0.0

    return {
        "retrieval": retrieval_metrics,
        "bertscore_lite": bs_lite,
        "bertscore_full": bs_full,
        "rouge": rouge,
    }


def demo_evaluation():
    """Demonstra todas as métricas com o conjunto de teste embutido."""
    from store import get_vector_store, SAMPLE_DOCS
    from pipeline import build_rag_chain, validate_and_answer

    print("=" * 62)
    print("  MÓDULO AVANÇADO — Avaliação do Pipeline RAG")
    print("=" * 62)

    store = get_vector_store()
    chain, _ = build_rag_chain(store)

    # Conjunto de teste com gabaritos
    TEST_SET = [
        {
            "query":         "Qual o prazo para entrega da dissertação?",
            "relevant_text": "prazo para entrega da dissertação de mestrado é de 24 meses",
            "reference":     "O prazo para entrega da dissertação de mestrado é de 24 meses após o ingresso.",
        },
        {
            "query":         "Horário de funcionamento da biblioteca",
            "relevant_text": "biblioteca universitária funciona de segunda a sexta",
            "reference":     "A biblioteca funciona de segunda a sexta das 8h às 22h, e sábados das 9h às 17h.",
        },
        {
            "query":         "Como funciona o Restaurante Universitário?",
            "relevant_text": "Restaurante Universitário",
            "reference":     "O RU oferece refeições subsidiadas para alunos matriculados.",
        },
    ]

    retrieval_fn  = lambda q: store.similarity_search(q, k=5)
    generation_fn = lambda q: validate_and_answer(q, chain, store)

    # --- MRR ---
    print("\n--- Retrieval: MRR@5 ---")
    mrr = mrr_at_k(TEST_SET, retrieval_fn, k=5)
    print(f"  MRR@5        : {mrr['mrr']}")
    print(f"  Recall@5     : {mrr['recall_at_k']}")
    print(f"  Precision@5  : {mrr['precision_at_k']}")
    for d in mrr["details"]:
        rank_str = f"rank={d['rank']}" if d["rank"] > 0 else "não encontrado"
        print(f"  [{rank_str}] '{d['query'][:45]}'")

    # --- BERTScore Lite ---
    print("\n--- Generation: BERTScore Lite (API, sem modelo local) ---")
    preds = [generation_fn(item["query"]) for item in TEST_SET]
    refs  = [item["reference"] for item in TEST_SET]
    bs = bertscore_lite(preds, refs)
    print(f"  F1 médio : {bs['f1_mean']}")
    print(f"  F1 por query: {bs['scores']}")

    # --- ROUGE-L ---
    print("\n--- Generation: ROUGE-L ---")
    for pred, ref, item in zip(preds, refs, TEST_SET):
        r = rouge_l_score(pred, ref)
        print(f"  F1={r.get('f1','?')} | '{item['query'][:45]}'")

    # --- LLM Judge (primeira query) ---
    print("\n--- Generation: LLM Judge (primeira query) ---")
    item = TEST_SET[0]
    docs = store.similarity_search(item["query"], k=3)
    ctx  = "\n".join(d.page_content for d in docs)
    judge = llm_judge(item["query"], preds[0], ctx, item["reference"])
    for k_score in ["fidelidade", "relevancia", "completude", "score_medio"]:
        if k_score in judge:
            print(f"  {k_score:<15}: {judge[k_score]}")
    if "justificativa" in judge:
        print(f"  justificativa  : {judge['justificativa']}")


# =============================================================================
# Execução direta
# python evaluation.py
# Pré-requisito: python store.py
# =============================================================================
if __name__ == "__main__":
    demo_evaluation()
