# SEC Analyst Agent

Agente que responde no Slack perguntas sobre empresas listadas nos EUA, a partir de relatórios 10-K/10-Q e números XBRL da SEC, sempre com citações. Projeto de aprendizado (LLM, RAG, agentes) e de portfólio. Spec: `docs/specs/2026-11-25-sec-analyst-agent.md`.

## Stack

Python 3.12 (uv, ruff, mypy strict, pytest), Postgres 16 + pgvector, Docker Compose. sentence-transformers com PyTorch CPU para embeddings locais, SDK da Anthropic para o agente, sqlglot para validar SQL, Slack Bolt em Socket Mode.

## Estrutura

- `common/`: settings (`pydantic-settings`) e conexões com o banco.
- `db/`: migrations SQL numeradas e o runner `db.migrate`.
- `ingestion/`: cliente da SEC, `xbrl/` (fatos brutos e métricas curadas), `filings/` (listagem, parsing em seções, persistência), `pipeline.py` (um passo por função) e `cli.py`.
- `rag/`: chunking, `embeddings/` (Protocol + implementação local), indexação em dois passos (chunk, embed), busca híbrida com RRF, re-ranker opcional e `cli.py`.
- `agent/`: loop de tool use escrito à mão (`loop.py`), `tools/`, validador de SQL, registro de fontes e citações, tracing em `agent_runs`/`agent_steps`, `factory.py` (monta as dependências) e `cli.py`.
- `slack_app/`: bot em Socket Mode. `main.py` só liga os eventos do Bolt; a lógica fica em `handler.py` (testável sem Slack), `formatting.py` (Markdown → blocos do Slack) e `history.py` (contexto da thread, lido de `agent_runs`).
- `evals/`: `dataset.yaml` (casos e gabaritos), eval de recuperação (sem LLM), eval de resposta (agente + juiz LLM) e `run.py`. Resultados vão para `data/evals/` (fora do git).
- `config/`: configuração versionada sem segredos (empresas e mapa métrica → tags XBRL).
- `docs/`: specs, design e `CONCEITOS.md` (pt-BR). O `README.md` é em inglês.

## Padrões

- Tudo roda no container `app` via `make` (`up`, `migrate`, `ingest`, `index`, `search`, `ask`, `slack`, `eval-retrieval`, `eval`, `test`, `lint`, `psql`); o código é montado como volume.
- Mudança de schema é sempre uma migration nova em `db/migrations/`; nunca editar uma já aplicada.
- Dois papéis no banco: admin (migrations, ingestão, tracing) e o login read-only do agente, com `SELECT` apenas na allow-list de fundamentos. Tabela ou view nova que o agente deva consultar precisa de `GRANT` explícito para `agent_readonly`.
- Ingestão é idempotente (upsert por chave natural) e escrita como funções puras de extract/transform/load, para virarem tasks do Airflow depois.
- Números têm duas camadas: `financial_facts` (bruto, como a SEC publica) e `financial_metrics` (um valor por empresa, métrica e período fiscal, reconstruída a cada ingestão). Regras de derivação ficam em `ingestion/xbrl/metrics.py`.
- Filing já ingerido é pulado; `--force` apaga e recria (o cascade remove seções e chunks).
- Dados da SEC nunca são escritos à mão no repositório (CIKs, tags, valores): vêm das APIs oficiais.
- Modelos (embedding, re-ranker) ficam atrás de um `Protocol` e são importados de forma tardia; os testes usam implementações falsas e nunca carregam PyTorch.
- O texto embedado é o chunk mais um cabeçalho de contexto (empresa, relatório, período, seção); a coluna `content` guarda só o chunk.
- O agente só fala com a API da Anthropic em `agent/llm.py`. A conversa é append-only: cada resposta do modelo volta para a API exatamente como veio.
- Tool nova implementa o Protocol `Tool`; erro que o modelo pode corrigir é `ToolInputError` (vira `tool_result` com `is_error`), qualquer outra exceção derruba o run.
- Fontes citáveis recebem id do `SourceRegistry`, nunca do modelo; a lista de fontes da resposta é montada pelo código.
- O prompt de sistema e as definições das tools precisam ser idênticos entre requisições (prompt caching): nada de data, id ou ordem variável ali.
- Textos que o usuário lê no Slack são em português e ficam como constantes em `slack_app/formatting.py`. O modelo escreve Markdown; a conversão para o mrkdwn do Slack é feita no código.
- Gabarito de eval é sempre lido dos dados ingeridos, nunca escrito de memória. Caso novo de texto precisa de `retrieval` (query, filtros, frase-chave); o teste `test_dataset_loads_and_is_well_formed` cobra isso.
- `make eval` gasta dinheiro (API): rodar só com aprovação do Robson.
- Testes de integração usam o marcador `integration` e rodam num banco próprio (`<POSTGRES_DB>_test`, criado pelo `tests/conftest.py`), nunca no de desenvolvimento.
