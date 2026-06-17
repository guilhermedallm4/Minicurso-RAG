# Melhorias no Crawler e Chunking

## 📋 Resumo das Alterações

Foram implementadas três melhorias principais para capturar dados mais estruturados e fazer chunking semanticamente coerente:

### 1. ✨ Chunking Semântico (`aplicacao/chunking_semantico.py`)

**Novo módulo** que reconhece e respeita a estrutura semântica dos documentos institucionais.

#### Características:
- **Reconhecimento de Seções**: Detecta automaticamente seções estruturadas como:
  - `Resumo:`, `Objetivos:`, `Justificativa:`, `Metodologia:`
  - `Ementa:`, `Conteúdo Programático:`, `Equipe:`
  - E mais 20+ padrões comuns no portal UFPel

- **Preservação de Coerência**: Mantém seções pequenas inteiras sem quebrar no meio
- **Agrupamento Inteligente**: Agrupa seções pequenas adjacentes para atingir tamanho mínimo
- **Metadados Enriquecidos**: Adiciona ao metadata de cada chunk:
  - `section_headers`: lista dos headers encontrados
  - `has_semantic_structure`: bool indicando se chunk tem estrutura reconhecida

#### Fluxo:
```
Documento bruto
  ↓
Extrai seções estruturadas [Resumo, Objetivos, Equipe, ...]
  ↓
Se tem múltiplas seções → agrupa inteligentemente mantendo coerência
  ↓
Se não tem estrutura → fallback para RecursiveCharacterTextSplitter
  ↓
Chunks com metadados enriquecidos
```

#### Exemplo de Saída:

```
Entrada:
Tipo: Projeto
Nome: Sistema de IA
Resumo: Lorem ipsum dolor sit amet... (200 chars)
Objetivos: Implementar algoritmos... (150 chars)
Equipe: - João Silva (150 chars)

Chunking Semântico (3 chunks coherentes):
[1] Tipo: Projeto
    Nome: Sistema de IA
    Resumo: Lorem ipsum... (250 chars)

[2] Objetivos: Implementar algoritmos... (150 chars)

[3] Equipe: - João Silva... (150 chars)

vs

Chunking Recursivo (pode quebrar seções):
[1] Tipo: Projeto
    Nome: Sistema de IA
    Resumo: Lorem ipsu... (500 chars - CORTA SEÇÃO)
```

---

### 2. 🎯 Extração de Equipe em Projetos

**Melhoria em `crawler/crawl_ufpel.py`**: Nova função `_extract_equipe_members()`

#### O que agora é capturado:
- **Antes**: Apenas texto em bloco das fichas principais
- **Depois**: 
  - Membros extraídos de tabelas estruturadas
  - Membros extraídos de listas (`<ul>`, `<li>`)
  - Membros extraídos de seções com ID/classe "equipe"
  - Papel/função de cada membro (coordenador, desenvolvedor, etc.)

#### Exemplo:
```html
<!-- Página do projeto contém -->
<div id="equipe">
  <table>
    <tr>
      <td>Dr. João Silva</td>
      <td>Coordenador</td>
    </tr>
    <tr>
      <td>Maria Santos</td>
      <td>Pesquisadora</td>
    </tr>
  </table>
</div>
```

**JSON agora captura:**
```json
{
  "url": "...",
  "text": "Tipo: Projeto\nNome: ...\nResumo: ...\nEquipe:\n- Dr. João Silva (Coordenador)\n- Maria Santos (Pesquisadora)\n..."
}
```

---

### 3. 📚 Resumo de Professores/Servidores

**Melhoria existente aprimorada**: O campo de `Resumo` (quando existe) agora é preservado estruturadamente

O crawler já capturava:
- Nome do Servidor
- Matrícula SIAPE
- Cargo
- Titulação
- Unidade/Departamento
- Função

Esses dados continuam sendo capturados no formato estruturado `Label: Valor`.

#### Informações de Servidor Capturadas:
```json
{
  "tipo": "servidor",
  "text": "Tipo: Servidor\nNome do Servidor: JOSÉ CARLOS SILVA\nMatrícula SIAPE: 1234567\nCargo: Professor do Magistério Superior\nTitulação: Doutorado\nFunção / Unidade: Professor / Departamento de Computação\n..."
}
```

---

## 🔧 Como Usar as Melhorias

### Para Fazer Crawling com Novas Funcionalidades:

```bash
# Crawl de projetos (com extração de equipe)
python crawler/crawl_ufpel.py --projects-only --projects-max 100

# Crawl de servidores (com informações estruturadas)
python crawler/crawl_ufpel.py --servidores-only

# Crawl de tudo (recomendado para usar todas as melhorias)
python crawler/crawl_ufpel.py --all-types --projects-max 500
```

### Para Ingerir com Chunking Semântico:

```bash
# Ingestão com CHUNKING SEMÂNTICO (padrão - RECOMENDADO)
python crawler/ingest_ufpel.py --input dados_ufpel.json --reset --chunking semantico

# Ingestão com chunking recursivo (para comparação)
python crawler/ingest_ufpel.py --input dados_ufpel.json --reset --chunking recursivo

# Com limites (free tier Google)
python crawler/ingest_ufpel.py --max-por-tipo 200 --delay 62 --chunking semantico
```

### Testar Chunking Isoladamente:

```bash
# Ver comparação entre os dois métodos
python test_semantic_chunking.py

# Teste direto do módulo
python aplicacao/chunking_semantico.py
```

---

## 📊 Impacto na Qualidade

### Antes das Melhorias:

```
Documento: Projeto com equipe de 5 pessoas
Chunking Recursivo: 8 chunks
  - Pode quebrar "Equipe:" no meio da lista
  - Perde contexto de que lista pertence a "Equipe"
  - Busca por "equipe do projeto" pode não encontrar

Documento: Disciplina com Ementa + Objetivos + Conteúdo
Chunking Recursivo: 12 chunks
  - Ementa pode ficar em chunks separados
  - Contexto "isto é Ementa" perdido nos chunks intermediários
```

### Depois das Melhorias:

```
Documento: Projeto com equipe de 5 pessoas
Chunking Semântico: 4 chunks (bem agrupados)
  - [1] Resumo completo
  - [2] Objetivos + Metodologia (pequenas, agrupadas)
  - [3] Equipe inteira (não quebrada)
  - Busca por "equipe" encontra contexto completo

Documento: Disciplina com Ementa + Objetivos + Conteúdo
Chunking Semântico: 3 chunks (estruturados)
  - [1] Identificação + Carga Horária
  - [2] Ementa (inteira)
  - [3] Objetivos + Conteúdo Programático
  - Cada chunk sabe qual "seção" pertence → melhor contexto
```

---

## 🏗️ Estrutura de Arquivos Modificados

```
minicurso_rag/
├── crawler/
│   ├── crawl_ufpel.py           ← MODIFICADO: _extract_equipe_members()
│   ├── ingest_ufpel.py          ← MODIFICADO: usa chunking_semantico
│   └── dados_ufpel.json         (entrada - sem mudanças, mas com mais dados)
│
├── aplicacao/
│   ├── chunking.py              (sem mudanças - ainda funciona)
│   ├── chunking_semantico.py    ← NOVO: chunking inteligente
│   └── config.py                (sem mudanças)
│
└── test_semantic_chunking.py    ← NOVO: script de teste comparativo
```

---

## 📋 Mapeamento de Seções Reconhecidas

O chunker semântico reconhece automaticamente:

### Identificação (todos os tipos):
- `Tipo:`, `Nome do Projeto:`, `Nome da Atividade:`, `Nome do Servidor:`, `Nome da Unidade:`, `Nome do Curso:`

### Descritivos:
- `Resumo:`, `Objetivo Geral:`, `Objetivos:`, `Justificativa:`, `Metodologia:`, `Ementa:`, `Conteúdo Programático:`

### Equipe e Recursos Humanos:
- `Equipe:`, `Membros da Equipe:`, `Coordenador:`, `Coordenador Atual:`, `Docentes:`, `Servidores:`

### Atributos:
- `Titulação:`, `Cargo:`, `Unidade:`, `Departamento:`, `Carga Horária:`, `Créditos:`, `Ênfase:`, `Área CNPq:`, `Linha de Extensão:`, `Eixo Temático:`, `Data inicial - Data final:`

---

## 🚀 Próximos Passos Sugeridos

1. **Executar Crawling Completo**:
   ```bash
   python crawler/crawl_ufpel.py --all-types --projects-max 500 --disciplines-max 500
   ```

2. **Ingerir com Chunking Semântico**:
   ```bash
   python crawler/ingest_ufpel.py --reset --chunking semantico
   ```

3. **Testar Qualidade** (opcional):
   ```bash
   python test_semantic_chunking.py
   ```

4. **Atualizar RAG** (automático quando usar novas coleções):
   - O pipeline RAG usa automaticamente metadados enriquecidos
   - Busca pode usar `section_headers` para filtros avançados

---

## ⚙️ Configurações Ajustáveis

Em `aplicacao/config.py`:

```python
CHUNK_SIZE    = 500   # Aumentar para chunks maiores (ex: 800)
CHUNK_OVERLAP = 50    # Aumentar para mais contexto entre chunks (ex: 100)
```

No novo chunker semântico (em `chunking_semantico.py`):

```python
def _merge_small_sections(sections, min_size=100):  # Seções < 100 chars são agrupadas
```

---

## 📝 Notas de Implementação

- **Backward Compatible**: Código antigo continua funcionando
- **Fall-back Seguro**: Se seção não for reconhecida, usa RecursiveCharacterTextSplitter
- **Metadados Preservados**: Todos os metadados originais são mantidos + novos campos adicionados
- **Determinístico**: Mesmo documento sempre gera mesmos chunks (sem randomização)

---

## ✅ Checklist de Testes Recomendados

- [ ] Executar `test_semantic_chunking.py` para validar
- [ ] Verificar alguns chunks no banco: `SELECT * FROM ufpel_projetos LIMIT 5;`
- [ ] Testar busca por seção: "encontre disciplinas de IA"
- [ ] Testar busca por pessoa: "quem coordena o projeto X"
- [ ] Verificar metadados: `SELECT metadata FROM langchain_pg_embedding LIMIT 5;`
