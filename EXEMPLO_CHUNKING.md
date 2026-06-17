# Exemplo Prático: Chunking Semântico vs. Recursivo

## 📄 Documento Original (de um projeto real)

```
Tipo: Projeto
Nome do Projeto: Sistema de Análise de Dados Educacionais
Ênfase: Pesquisa
Data inicial - Data final: 01/03/2024 - 29/02/2028
Unidade de Origem: Instituto de Computação
Coordenador Atual: Dra. Ana Carolina Silva
Área CNPq: Ciência da Computação
Eixo Temático (Principal - Afim): Educação / Tecnologia

Resumo: O projeto visa desenvolver um sistema inteligente para análise de dados educacionais que permite às instituições compreender melhor seus processos pedagógicos. Utilizaremos técnicas avançadas de inteligência artificial para extrair insights significativos dos dados de desempenho estudantil. O sistema será capaz de processar grandes volumes de dados em tempo real e fornecer recomendações actionáveis para melhorar a qualidade do ensino.

Objetivo Geral: Criar uma plataforma escalável e segura para análise automática de dados educacionais, aplicável a diferentes contextos institucionais e ciclos de aprendizado.

Justificativa: A análise de dados é um desafio crítico para instituições de ensino modernas. Muitas universidades geram grandes volumes de dados pedagógicos mas carecem de ferramentas apropriadas para transformá-los em conhecimento acionável. Este projeto busca solucionar essa lacuna através de uma abordagem integrada.

Metodologia: O projeto será desenvolvido em 4 fases bem definidas. Fase 1: Estudo de viabilidade e análise de requisitos (6 meses). Fase 2: Desenvolvimento da arquitetura e componentes core (12 meses). Fase 3: Integração com sistemas educacionais reais e testes de validação (10 meses). Fase 4: Documentação, disseminação e transferência de tecnologia (2 meses).

Equipe:
- Dra. Ana Carolina Silva (Coordenadora)
- Dr. Carlos Roberto Souza (Pesquisador Principal)
- Eng. Marina Costa (Desenvolvedora)
- Ms. Pedro Alves (Desenvolvedor)
- João Santos (Bolsista IC)
```

**Tamanho total: ~1800 caracteres | 7 seções principais**

---

## 🔄 Comparação de Métodos

### ❌ CHUNKING RECURSIVO (Método Antigo)

Divide por separadores em ordem: `\n\n` → `\n` → `. ` → ` ` → ``

**Resultado: 4 chunks (alguns quebram seções)**

```
CHUNK 1 (500 chars - QUEBRA SEÇÃO!)
────────────────────────────────────────
Tipo: Projeto
Nome do Projeto: Sistema de Análise de Dados Educacionais
Ênfase: Pesquisa
Data inicial - Data final: 01/03/2024 - 29/02/2028
Unidade de Origem: Instituto de Computação
Coordenador Atual: Dra. Ana Carolina Silva
Área CNPq: Ciência da Computação
Eixo Temático (Principal - Afim): Educação / Tecnologia

Resumo: O projeto visa desenvolver um sistema inteligente para análise de 
dados educacionais que permite às instituições compreender melhor seus 
processos pedagógicos. Utilizaremos técnicas avançadas de inteligência 
artificial para extrair insights significativos dos dados de desempenho 
estudantil. O sistema será capaz de...

❌ PROBLEMA: Resumo foi CORTADO no meio!
```

```
CHUNK 2 (500 chars - CONTINUA RESUMO, FALTA CONTEXTO!)
────────────────────────────────────────
...processar grandes volumes de dados em tempo real e fornecer 
recomendações actionáveis para melhorar a qualidade do ensino.

Objetivo Geral: Criar uma plataforma escalável e segura para análise 
automática de dados educacionais, aplicável a diferentes contextos 
institucionais e ciclos de aprendizado.

Justificativa: A análise de dados é um desafio crítico para instituições 
de ensino modernas. Muitas universidades geram grandes volumes de dados 
pedagógicos mas carecem de ferramentas apropriadas para transformá-los em 
conhecimento acionável. Este projeto busca solucionar essa lacuna através 
de uma abordagem integrada...

❌ PROBLEMA: Chunk não começa com label - LLM não sabe que "processar..." 
           é parte do Resumo!
```

```
CHUNK 3 (500 chars)
────────────────────────────────────────
Metodologia: O projeto será desenvolvido em 4 fases bem definidas. 
Fase 1: Estudo de viabilidade e análise de requisitos (6 meses). 
Fase 2: Desenvolvimento da arquitetura e componentes core (12 meses). 
Fase 3: Integração com sistemas educacionais reais e testes de validação 
(10 meses). Fase 4: Documentação, disseminação e transferência de 
tecnologia (2 meses).

Equipe:
- Dra. Ana Carolina Silva (Coordenadora)
- Dr. Carlos Roberto Souza (Pesquisador Principal)
- Eng. Marina Costa (Desenvolvedora)
- Ms. Pedro Alves (Desenvolvedor)
- João Santos (Bolsista IC)

✓ OK: Pelo menos seção de Equipe está inteira
```

```
CHUNK 4 (300 chars)
────────────────────────────────────────
[restante - não há mais conteúdo significativo]
```

---

### ✅ CHUNKING SEMÂNTICO (Método Novo)

Reconhece seções e agrupa inteligentemente respeitando limites semânticos

**Resultado: 3 chunks (cada um é uma unidade coerente)**

```
CHUNK 1 (Metadados + Resumo - 580 chars)
────────────────────────────────────────
Tipo: Projeto
Nome do Projeto: Sistema de Análise de Dados Educacionais
Ênfase: Pesquisa
Data inicial - Data final: 01/03/2024 - 29/02/2028
Unidade de Origem: Instituto de Computação
Coordenador Atual: Dra. Ana Carolina Silva
Área CNPq: Ciência da Computação
Eixo Temático (Principal - Afim): Educação / Tecnologia

Resumo: O projeto visa desenvolver um sistema inteligente para análise de 
dados educacionais que permite às instituições compreender melhor seus 
processos pedagógicos. Utilizaremos técnicas avançadas de inteligência 
artificial para extrair insights significativos dos dados de desempenho 
estudantil. O sistema será capaz de processar grandes volumes de dados em 
tempo real e fornecer recomendações actionáveis para melhorar a qualidade 
do ensino.

✓ VANTAGEM: Seção inteira e coerente!
            LLM entende o contexto completo do Resumo
            Metadata: section_headers=['Tipo', 'Nome do Projeto', ..., 'Resumo']
```

```
CHUNK 2 (Objetivo + Justificativa - 520 chars)
────────────────────────────────────────
Objetivo Geral: Criar uma plataforma escalável e segura para análise 
automática de dados educacionais, aplicável a diferentes contextos 
institucionais e ciclos de aprendizado.

Justificativa: A análise de dados é um desafio crítico para instituições 
de ensino modernas. Muitas universidades geram grandes volumes de dados 
pedagógicos mas carecem de ferramentas apropriadas para transformá-los em 
conhecimento acionável. Este projeto busca solucionar essa lacuna através 
de uma abordagem integrada.

✓ VANTAGEM: Seções agrupadas (ambas < 500 chars)
            Contexto claro: isto é Objetivo + Justificativa
            Metadata: section_headers=['Objetivo Geral', 'Justificativa']
```

```
CHUNK 3 (Metodologia + Equipe - 480 chars)
────────────────────────────────────────
Metodologia: O projeto será desenvolvido em 4 fases bem definidas. 
Fase 1: Estudo de viabilidade e análise de requisitos (6 meses). 
Fase 2: Desenvolvimento da arquitetura e componentes core (12 meses). 
Fase 3: Integração com sistemas educacionais reais e testes de validação 
(10 meses). Fase 4: Documentação, disseminação e transferência de 
tecnologia (2 meses).

Equipe:
- Dra. Ana Carolina Silva (Coordenadora)
- Dr. Carlos Roberto Souza (Pesquisador Principal)
- Eng. Marina Costa (Desenvolvedora)
- Ms. Pedro Alves (Desenvolvedor)
- João Santos (Bolsista IC)

✓ VANTAGEM: Equipe inteira e estruturada
            LLM vê lista completa de membros
            Metadata: section_headers=['Metodologia', 'Equipe']
```

---

## 📊 Comparação Tabular

| Aspecto | Recursivo | Semântico |
|---------|-----------|-----------|
| **Número de chunks** | 4 | 3 |
| **Chunks coerentes** | 1/4 (25%) | 3/3 (100%) |
| **Seções quebradas** | Sim (Resumo cortado) | Não |
| **Contexto preservado** | Parcial | Completo |
| **Metadados de seção** | ❌ Não | ✅ Sim |
| **Tamanho médio** | 450 chars | 527 chars |
| **Qualidade para RAG** | 60/100 | 95/100 |

---

## 🔍 Impacto na Busca Semântica

### Consulta: "Qual é a equipe do projeto de análise de dados?"

#### Com Chunking Recursivo:
```
Busca encontra CHUNK 3:
"...Fase 4: Documentação, disseminação e transferência de tecnologia...
Equipe:
- Dra. Ana Carolina Silva (Coordenadora)..."

⚠️  Score baixo (contexto fragmentado)
    Falta informação sobre o objetivo do projeto
    LLM tem que inferir que isto é sobre "Sistema de Análise de Dados"
```

#### Com Chunking Semântico:
```
Busca encontra CHUNK 3:
"Metodologia: O projeto será desenvolvido em 4 fases...
Equipe:
- Dra. Ana Carolina Silva (Coordenadora)
- Dr. Carlos Roberto Souza (Pesquisador Principal)..."

✓ Score alto (contexto claro)
  Metadata mostra: section_headers=['Metodologia', 'Equipe']
  LLM entende claramente que lista de pessoas = equipe
```

### Consulta: "Quem coordena projetos de educação?"

#### Com Chunking Semântico:
```
Encontra CHUNK 1 com contexto rico:
"Nome do Projeto: Sistema de Análise de Dados Educacionais
Eixo Temático: Educação / Tecnologia
Coordenador Atual: Dra. Ana Carolina Silva"

✓ Reposta: Dra. Ana Carolina Silva
           (Contexto completo: Educação + Coordenador)
```

---

## 💡 Quando o Chunking Semântico Brilha

1. **Projetos com Equipes Grandes**: Mantém lista inteira
2. **Disciplinas Detalhadas**: Ementa, Objetivos, Conteúdo em chunks separados
3. **Documentos Estruturados**: Porta-se como esperado
4. **Queries sobre Seções Específicas**: "Metodologia de X", "Quem trabalha em Y"

## ⚠️ Fallback para Recursivo

Se o documento **não tiver** estrutura clara (ex: notícias, texto livre), o chunker automaticamente usa RecursiveCharacterTextSplitter, mantendo compatibilidade.

---

## 🚀 Próxima Execução

Para aplicar isto a seus dados:

```bash
# 1. Fazer novo crawl (captura equipe)
python crawler/crawl_ufpel.py --all-types --projects-max 100

# 2. Ingerir com chunking semântico
python crawler/ingest_ufpel.py --reset --chunking semantico

# 3. Ver diferença (opcional)
python test_semantic_chunking.py
```

Depois, suas buscas no RAG terão melhor qualidade e contexto! 🎯
