"""
Chunking Semântico com Reconhecimento de Seções
=================================================
Estratégia avançada de segmentação que reconhece a estrutura semântica dos
documentos institucionais (Resumo, Objetivos, Equipe, etc.) e mantém seções
coerentes juntas ao fazer chunks.

Diferentemente do RecursiveCharacterTextSplitter que apenas respeita separadores
de caracteres, este módulo:
  1. Identifica seções estruturadas (Resumo:, Objetivos:, Equipe:, etc.)
  2. Mantém seções pequenas inteiras (sem quebrar no meio)
  3. Agrupa seções pequenas adjacentes para atingir tamanho mínimo
  4. Adiciona metadados sobre qual seção cada chunk pertence
"""
import re
from typing import List, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# Padrões de seções comuns em documentos UFPel
SECTION_HEADERS = [
    "Tipo:",
    "Nome do Projeto:",
    "Nome da Atividade:",
    "Nome do Servidor:",
    "Nome da Unidade:",
    "Nome do Curso:",
    "Resumo:",
    "Objetivo Geral:",
    "Objetivos:",
    "Justificativa:",
    "Metodologia:",
    "Ementa:",
    "Conteúdo Programático:",
    "Equipe:",
    "Membros da Equipe:",
    "Coordenador:",
    "Coordenador Atual:",
    "Docentes:",
    "Servidores:",
    "Titulação:",
    "Cargo:",
    "Unidade:",
    "Departamento:",
    "Carga Horária:",
    "Créditos:",
    "Ênfase:",
    "Área CNPq:",
    "Linha de Extensão:",
    "Eixo Temático:",
    "Data inicial - Data final:",
]

# Regexes para encontrar seções
SECTION_PATTERN = r"^(" + "|".join(re.escape(h) for h in sorted(SECTION_HEADERS, key=len, reverse=True)) + r")\s"


@dataclass
class Section:
    """Representa uma seção estruturada do documento."""
    header: str
    content: str
    start_char: int
    end_char: int

    @property
    def text(self) -> str:
        """Retorna header + content."""
        return f"{self.header} {self.content}".strip()

    @property
    def size(self) -> int:
        """Tamanho em caracteres."""
        return len(self.text)


def _extract_sections(text: str) -> List[Section]:
    """
    Extrai seções estruturadas do texto.

    Estratégia:
      1. Procura por padrões de header (Resumo:, Objetivos:, etc.)
      2. Agrupa conteúdo até o próximo header
      3. Retorna lista de Section com boundaries
    """
    lines = text.split("\n")
    sections = []
    current_header = None
    current_content_lines = []
    start_pos = 0

    for line_idx, line in enumerate(lines):
        # Detecta um novo header
        match = re.match(SECTION_PATTERN, line)
        if match:
            # Salva seção anterior se existir
            if current_header is not None:
                content = "\n".join(current_content_lines).strip()
                sections.append(Section(
                    header=current_header,
                    content=content,
                    start_char=start_pos,
                    end_char=start_pos + len(f"{current_header}\n{content}"),
                ))

            current_header = match.group(1)
            current_content_lines = [line[len(current_header):].strip()]
            start_pos = sum(len(lines[i]) + 1 for i in range(line_idx))
        else:
            # Continua acumulando conteúdo da seção atual
            if current_header is not None:
                current_content_lines.append(line)
            elif sections:  # Ignora linhas antes de qualquer header
                pass

    # Salva última seção
    if current_header is not None:
        content = "\n".join(current_content_lines).strip()
        sections.append(Section(
            header=current_header,
            content=content,
            start_char=start_pos,
            end_char=start_pos + len(f"{current_header}\n{content}"),
        ))

    return sections


def _merge_small_sections(
    sections: List[Section],
    min_size: int = 100,
) -> List[str]:
    """
    Agrupa seções muito pequenas com adjacentes para atingir tamanho mínimo.
    Retorna lista de textos (chunks) já agrupados.
    """
    if not sections:
        return []

    chunks = []
    current_chunk_sections = [sections[0]]
    current_size = sections[0].size

    for section in sections[1:]:
        # Se adicionar esta seção mantém tamanho razoável, agrupa
        if current_size + section.size <= config.CHUNK_SIZE:
            current_chunk_sections.append(section)
            current_size += section.size
        else:
            # Finaliza chunk atual
            chunk_text = "\n\n".join(s.text for s in current_chunk_sections)
            chunks.append(chunk_text)

            # Inicia novo chunk
            current_chunk_sections = [section]
            current_size = section.size

    # Finaliza último chunk
    if current_chunk_sections:
        chunk_text = "\n\n".join(s.text for s in current_chunk_sections)
        chunks.append(chunk_text)

    return chunks


def chunk_documents_semantic(documents: List[Document]) -> List[Document]:
    """
    Divide Documents do LangChain em chunks respeitando estrutura semântica.

    Estratégia de 3 fases:
      1. Tenta extrair seções estruturadas (Resumo:, Objetivos:, etc.)
      2. Se encontrar seções, agrupa as pequenas e cria chunks mantendo
         integridade semântica
      3. Se não encontrar seções, fall back para RecursiveCharacterTextSplitter

    Metadados adicionados:
      - section_headers: lista dos headers encontrados no chunk
    """
    chunks = []
    skipped = 0

    for doc in documents:
        text = doc.page_content

        # Tenta extrair seções estruturadas
        sections = _extract_sections(text)

        if sections and len(sections) > 1:
            # Temos estrutura — agrupa seções respeitando tamanho
            section_chunks = _merge_small_sections(sections)

            for chunk_text in section_chunks:
                # Extrai headers presentes neste chunk
                section_headers = []
                for line in chunk_text.split("\n"):
                    match = re.match(SECTION_PATTERN, line)
                    if match:
                        section_headers.append(match.group(1).rstrip(":"))

                metadata = doc.metadata.copy()
                metadata["section_headers"] = section_headers
                metadata["has_semantic_structure"] = True

                chunks.append(Document(page_content=chunk_text, metadata=metadata))
        else:
            # Sem estrutura clara ou documento muito curto — use splitter recursivo
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            sub_chunks = splitter.split_documents([doc])
            for chunk in sub_chunks:
                chunk.metadata["has_semantic_structure"] = False
            chunks.extend(sub_chunks)

    print(f"[Chunking Semântico] {len(documents)} doc(s) → {len(chunks)} chunks "
          f"(min_estruturado=100, max={config.CHUNK_SIZE})")
    return chunks


def chunk_texts_semantic(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
) -> List[Document]:
    """Converte strings brutas em Documents e aplica chunking semântico."""
    metadatas = metadatas or [{}] * len(texts)
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    return chunk_documents_semantic(docs)


# =============================================================================
# Execução direta: demonstra o efeito do chunking semântico
# =============================================================================
if __name__ == "__main__":
    TEXTO_EXEMPLO = """
Tipo: Projeto
Nome do Projeto: Sistema Inteligente para Análise de Dados
Coordenador Atual: DR. JOÃO SILVA
Área CNPq: Ciência da Computação
Ênfase: Pesquisa

Resumo: Este projeto visa desenvolver um sistema de análise de dados utilizando técnicas de inteligência artificial. O sistema será capaz de processar grandes volumes de dados e extrair insights significativos através de algoritmos de aprendizado de máquina.

Objetivo Geral: Criar uma plataforma robusta e escalável para análise automática de dados em tempo real, aplicável a diversos domínios institucionais.

Justificativa: A análise de dados é um desafio crítico para instituições modernas. Este projeto busca solucionar esse problema através de uma abordagem integrada e flexível.

Metodologia: O projeto será desenvolvido em fases. Na primeira fase, será realizado um estudo de viabilidade. Na segunda fase, será implementado o sistema core. Na terceira fase, será realizado testes de validação.

Equipe:
- Coordenador: Dr. João Silva
- Desenvolvedores: Maria Santos, Pedro Costa
- Pesquisadores: Ana Oliveira, Carlos Ferreira
""".strip()

    print("=" * 70)
    print("  CHUNKING SEMÂNTICO — Reconhecimento de Seções Estruturadas")
    print("=" * 70)

    print("\n--- Texto original ---")
    print(f"  {len(TEXTO_EXEMPLO)} caracteres\n")

    print("--- Seções detectadas ---")
    sections = _extract_sections(TEXTO_EXEMPLO)
    for i, sec in enumerate(sections, 1):
        print(f"  [{i}] {sec.header:<30} ({sec.size} chars)")
        print(f"      {sec.content[:80].replace(chr(10), ' ')}...")

    print("\n--- Chunking semântico ---")
    docs = chunk_texts_semantic(
        [TEXTO_EXEMPLO],
        [{"source": "projeto_ufpel.txt", "tipo": "projeto"}],
    )
    for i, d in enumerate(docs, 1):
        headers = d.metadata.get("section_headers", [])
        print(f"  Chunk {i}: {len(d.page_content)} chars | headers={headers}")
        print(f"    {d.page_content[:80].replace(chr(10), ' ')}...")
