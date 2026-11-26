# Spec: SEC Analyst Agent

## Objetivo

Agente que responde, em thread no Slack da TBDC, perguntas sobre empresas listadas nos EUA, usando relatórios 10-K/10-Q e números XBRL da SEC. Toda resposta traz citações (empresa, tipo de documento, data, seção); quando a informação não está nos dados, o agente diz que não encontrou.

O projeto serve a dois fins: aprender LLM, RAG, embeddings, banco vetorial e agentes na prática, e ser peça de portfólio.

## Escopo

- Empresas: NKE, AAPL, MSFT, KO, TSLA, configuráveis em `config/companies.yaml`.
- Textos: 10-K e 10-Q dos últimos 3 anos fiscais.
- Números: todo o histórico da Company Facts API para essas empresas.
- Tudo local, via Docker Compose. Sem Airflow no Compose por ora.

## Fontes

| Dado | Origem | Formato |
|---|---|---|
| Mapa ticker → CIK | arquivo oficial de tickers da SEC | JSON |
| Números | `data.sec.gov/api/xbrl/companyfacts` | JSON, um arquivo por empresa |
| Lista de filings | `data.sec.gov/submissions` | JSON |
| Texto dos relatórios | arquivo EDGAR (documento principal do filing) | HTML |

O `User-Agent` com contato, exigido pela SEC, vem do `.env`, e o cliente HTTP respeita o rate limit publicado.

## Arquitetura

- **Postgres + pgvector:** fundamentos relacionais (`companies`, `financial_facts`), texto (`filings`, `filing_sections`, `chunks` com `vector(384)`, HNSW e `tsvector`/GIN) e observabilidade (`agent_runs`, `agent_steps`).
- **Ingestão:** funções puras de extract, transform e load, chamadas por um CLI idempotente. As mesmas funções viram tasks de DAG numa fase posterior.
- **Embeddings:** `BAAI/bge-small-en-v1.5` local, atrás de um `Protocol` que permite trocar de provedor.
- **Busca:** vetor (cosseno) mais full-text, fundidos por Reciprocal Rank Fusion, com filtros por ticker, ano fiscal e tipo de documento. Re-ranker (cross-encoder) opcional.
- **Agente:** loop de tool use escrito à mão sobre a API da Anthropic, com três tools: `search_filings`, `query_financials` e `calculator`.
- **Slack:** Bolt for Python em Socket Mode.
- **Evals:** cerca de 20 perguntas; métricas de retrieval (recall@k, MRR) e de resposta (correção e presença de citação, por juiz LLM), incluindo perguntas sem resposta nos dados.

## Guardrails

- SQL do agente em três camadas: login com `SELECT` apenas nas tabelas de fundamentos; validação do SQL antes de executar (um único `SELECT`, tabelas permitidas); transação read-only por padrão, `statement_timeout` e limite de linhas.
- Prompt exige que toda afirmação venha de resultado de tool e que a resposta liste as fontes.
- Segredos só no `.env`, que é gitignored.

## Pontos de atenção nos dados

- Ano fiscal difere do calendário (o da Nike termina em maio): períodos usam os campos fiscais do XBRL.
- O 10-K traz o ano cheio; o quarto trimestre isolado é derivado.
- As tags XBRL variam por empresa: um mapa de conceitos (receita, custo, lucro bruto) é validado contra os dados reais na Fase 2.

## Fases

1. Infra: Compose, schema, migrations, role read-only.
2. Ingestão: números (XBRL) e textos (10-K/10-Q).
3. RAG: chunking, embeddings, busca híbrida, CLI de busca.
4. Agente: loop, tools, citações, tracing, CLI.
5. Slack: bot e guia de criação do app.
6. Evals, README final, diagramas, `docs/CONCEPTS.md` consolidado.

Cada fase termina com instruções de teste e espera confirmação antes da seguinte.
