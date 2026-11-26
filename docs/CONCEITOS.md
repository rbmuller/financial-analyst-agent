# Conceitos

Explicações dos conceitos usados no projeto, com exemplos tirados dele.

Os exemplos de busca usam a pergunta `Why did gross margin decline?` filtrada para a Nike, ano fiscal 2026. Para reproduzir:

```bash
make search Q="Why did gross margin decline?" ARGS="--ticker NKE --year 2026 --top-k 4 --mode vector"
```

## Embedding

Um embedding é uma lista de números que representa o significado de um texto. Textos com significado parecido viram listas parecidas, mesmo sem palavras em comum. O modelo usado aqui (`BAAI/bge-small-en-v1.5`, rodando local) devolve 384 números para qualquer texto.

Por dentro, o modelo quebra o texto em tokens (pedaços de palavra, de um vocabulário fixo de cerca de 30 mil), dá a cada token um vetor inicial e passa tudo por camadas que misturam os tokens entre si. É essa mistura que faz "margin" em "gross margin declined" e em "margin of error" terem representações diferentes. No fim, o texto inteiro é resumido em um único vetor.

O modelo só converte texto em vetor. Quem guarda o vetor é o nosso código (`rag/indexing.py`), na mesma linha da tabela `chunks` de onde o texto saiu.

Dois detalhes de `rag/embeddings/local.py`:

- **Pergunta e documento são embedados de forma diferente.** Os modelos BGE foram treinados com uma instrução fixa no início da pergunta (`Represent this sentence for searching relevant passages: `), e nada no início do documento. Por isso a interface tem `embed_query` e `embed_documents`.
- **Os vetores são normalizados** (comprimento 1), o que deixa a comparação por cosseno estável.

A interface `EmbeddingProvider` (`rag/embeddings/base.py`) isola o resto do código do modelo: trocar por um provedor pago é escrever uma classe nova.

## Banco vetorial

Um banco vetorial guarda embeddings e responde à pergunta "quais vetores estão mais perto deste?". Aqui é o próprio Postgres, com a extensão pgvector: a tabela `chunks` tem uma coluna `embedding vector(384)` ao lado do texto e dos metadados.

Manter tudo num banco só tem uma vantagem prática: a mesma query filtra por `ticker = 'NKE'` e ordena por proximidade semântica, sem sincronizar dois sistemas.

A proximidade é medida por distância de cosseno, que compara a direção dos vetores e ignora o tamanho. No pgvector é o operador `<=>`:

```sql
SELECT '[1,0]'::vector <=> '[0,1]'::vector;  -- 1 (direções ortogonais, nada em comum)
SELECT '[1,0]'::vector <=> '[2,0]'::vector;  -- 0 (mesma direção, mesmo significado)
```

## Índice HNSW

Sem índice, achar os vizinhos mais próximos exige calcular a distância para todos os chunks. O HNSW (Hierarchical Navigable Small World) evita isso com um grafo em camadas: as de cima têm poucos pontos e ligações longas, as de baixo têm todos os pontos e ligações curtas. A busca entra por cima, salta até a região certa e desce refinando.

O resultado é aproximado: o índice pode deixar de fora um vizinho verdadeiro em troca de velocidade. Para os 7,5 mil chunks deste projeto a busca exata também seria rápida; o índice está aqui porque é o que se usa quando o volume cresce.

```sql
CREATE INDEX chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
```

O `vector_cosine_ops` precisa combinar com o operador usado na query (`<=>`); com outro operador o Postgres ignora o índice.

**Armadilha com filtros.** O índice devolve os vizinhos mais próximos do banco inteiro, e o `WHERE ticker = 'NKE'` é aplicado depois. Se os 100 vizinhos mais próximos forem todos de outra empresa, a busca filtrada volta vazia mesmo havendo chunks da Nike. A solução é o *iterative scan* do pgvector (versão 0.8 em diante), que continua percorrendo o grafo até juntar resultados suficientes que passem no filtro. Está ligado em `rag/search.py` (`SET LOCAL hnsw.iterative_scan`) e coberto pelo teste `test_filters_are_applied_inside_the_vector_scan`, que monta exatamente esse cenário.

## Chunking

Chunking é dividir cada documento em pedaços menores antes de gerar embeddings. Há dois motivos:

- o modelo lê no máximo 512 tokens, e o MD&A de um 10-K da Nike tem cerca de 70 mil caracteres;
- um vetor só para um texto longo fica vago demais para casar com uma pergunta específica.

A estratégia em `rag/chunking.py`:

1. O relatório já foi dividido por seção na ingestão (Item 7, Item 1A etc.), então nenhum chunk mistura seções.
2. Dentro da seção, parágrafos inteiros são agrupados até 400 tokens. Um parágrafo maior que isso é cortado por frase.
3. Chunks vizinhos compartilham até 60 tokens (*overlap*), para que uma ideia cortada na fronteira apareça inteira em pelo menos um deles.

Os 60 relatórios viraram 7.542 chunks, com média de 330 tokens.

**Cabeçalho de contexto.** Um chunk como "Gross margin contraction of approximately 200 basis points" não diz de quem nem quando. Antes de gerar o embedding, o código prefixa uma linha de contexto:

```
NIKE, Inc. (NKE) 10-Q FY2026 Q3 | Part I, Item 2: Management's Discussion and Analysis
Reported EBIT decreased 4% reflecting higher reported revenues more than offset by: ...
```

O cabeçalho entra no vetor; a coluna `content` guarda só o texto original, que é o que aparece nas citações.

## Busca híbrida

São duas buscas com forças complementares, combinadas num resultado só.

- **Busca vetorial** acha trechos com o mesmo significado da pergunta, mesmo com outras palavras. A pergunta fala em "decline", e o primeiro resultado vetorial fala em "Gross margin contraction of approximately 700 basis points".
- **Busca por palavra-chave** (full-text do Postgres, coluna `content_tsv` com índice GIN) acha trechos com os termos exatos. É melhor para nomes de produto, siglas e números, que o embedding tende a borrar.

Os termos da pergunta são combinados com OU, não com E: uma pergunta quase nunca aparece palavra por palavra num relatório, e o ranking já favorece os trechos que cobrem mais termos.

**Reciprocal Rank Fusion (RRF).** As duas buscas devolvem notas em escalas diferentes (distância de cosseno e rank de full-text), que não dá para somar. O RRF ignora as notas e usa só a posição: cada trecho ganha `1 / (60 + posição)` em cada lista, e as parcelas são somadas.

No exemplo, os dois primeiros resultados híbridos ficaram quase empatados:

```
1º: vetor #12, palavra-chave #1   ->  1/(60+12) + 1/(60+1)  = 0,0139 + 0,0164 = 0,0303
2º: vetor #1,  palavra-chave #13  ->  1/(60+1)  + 1/(60+13) = 0,0164 + 0,0137 = 0,0301
```

O primeiro é a tabela de margem bruta do 10-Q seguida da explicação ("Consolidated gross margin was 300 basis points lower than the prior year primarily due to..."), que contém os termos exatos da pergunta. O segundo é o melhor resultado semântico. A busca híbrida traz os dois para o topo; cada busca sozinha deixaria um deles em 12º ou 13º.

## Re-ranking

Re-ranking é uma segunda etapa que reordena os candidatos da busca com um modelo mais preciso e mais lento.

O modelo de embedding é um *bi-encoder*: gera o vetor da pergunta e o vetor de cada trecho separadamente, sem nunca ver os dois juntos. Isso permite calcular os vetores dos trechos uma vez só, na indexação. O re-ranker é um *cross-encoder*: lê a pergunta e o trecho juntos e devolve uma nota de relevância. É mais preciso, mas precisa rodar uma vez por candidato a cada pergunta, então só é aplicado aos cerca de 40 trechos que a busca híbrida já selecionou.

No exemplo, com `--rerank` o primeiro lugar passou a ser um trecho que era o 7º da busca vetorial e o 22º em palavra-chave, e que dá a causa de forma direta:

> Gross margin contraction of 330 basis points, primarily due to new tariffs, partially offset by lower warehousing and logistics costs due to channel mix, and lower inventory obsolescence reserves.

Nem tudo melhora: o terceiro lugar virou um trecho genérico de Risk Factors, que fala de margem em tese e não do que aconteceu.

O custo: a busca passou de 0,04s para 0,7s em CPU. O re-ranker usado (`cross-encoder/ms-marco-MiniLM-L-6-v2`) é pequeno e foi treinado em buscas da web, não em texto financeiro; se ele compensa aqui é uma pergunta para os evals (seção abaixo), e por isso ele é opcional.

## LLM

Um LLM (large language model) é um modelo que recebe texto e gera texto. Aqui é o Claude, acessado pela API da Anthropic. Três propriedades moldam o projeto inteiro:

- **Não tem memória entre chamadas.** Cada requisição leva a conversa inteira de novo. O "estado" do agente é a lista `messages` em `agent/loop.py`.
- **Só sabe o que viu no treino e o que está na requisição.** Ele não conhece o 10-Q que a Nike publicou no mês passado, a menos que o texto seja colocado na conversa.
- **Gera texto plausível, não verificado.** Perguntado sobre um número que não tem, pode produzir um com aparência correta. Isso se chama alucinação, e é o que RAG, tools e guardrails existem para conter.

A cobrança é por token (pedaço de palavra), com preço diferente para entrada e saída. A pergunta de exemplo sobre a margem da Nike consumiu 65 mil tokens de entrada e 3,2 mil de saída, e custou US\$ 0,17.

**Prompt caching.** A cada volta do loop a conversa inteira é reenviada, então os mesmos tokens seriam pagos várias vezes. Com o cache, o trecho inicial que não mudou (definição das tools, prompt de sistema, voltas anteriores) custa uma fração do preço. O cache compara o início da requisição byte a byte: por isso o prompt de sistema em `agent/prompts.py` não tem data nem nada que varie.

## RAG

RAG (retrieval-augmented generation) é buscar os trechos relevantes primeiro e entregá-los ao LLM junto com a pergunta, para que ele responda a partir deles e não da memória.

Neste projeto são as três fases anteriores trabalhando juntas:

1. **Indexação (antes, uma vez):** relatórios → seções → chunks → embeddings no Postgres.
2. **Recuperação (a cada pergunta):** a busca híbrida devolve os 6 a 8 chunks mais relevantes.
3. **Geração:** o Claude recebe esses chunks e escreve a resposta, citando de qual chunk veio cada afirmação.

O RAG resolve duas limitações do LLM de uma vez: dá acesso a documentos que ele nunca viu, e torna a resposta verificável, porque cada afirmação aponta para um trecho que existe.

## Tool use

Tool use é o mecanismo pelo qual o LLM pede que o nosso código execute algo. O modelo nunca executa nada: ele devolve um pedido estruturado, e o programa decide se e como atender.

Cada tool tem duas faces (veja `agent/tools/base.py`):

- **Para o modelo:** nome, descrição e um JSON Schema dos argumentos. Esse texto é tudo o que o modelo sabe sobre a tool, então a descrição diz quando usar, e não só o que ela faz.
- **Para o código:** uma função Python comum.

Na pergunta de exemplo, o modelo devolveu este pedido em vez de texto:

```json
{"type": "tool_use", "name": "calculator",
 "input": {"operation": "ratio_percent", "values": [4530000000, 11279000000]}}
```

O código executou a conta e devolveu `{"result": 40.1631}` numa mensagem `tool_result`. Só então o modelo escreveu "40,2%".

As três tools do projeto:

| Tool | Para quê | Por que existe |
|---|---|---|
| `search_filings` | Trechos dos relatórios (o RAG) | O que a diretoria escreveu não está em tabela nenhuma |
| `query_financials` | SQL sobre os números do XBRL | Busca semântica é ruim para números exatos; SQL é exato |
| `calculator` | Variação %, CAGR, média, razão | LLM prevê dígitos plausíveis, não calcula |

## Agente

Um agente é um LLM dentro de um loop, em que o próprio modelo decide o próximo passo. O loop inteiro está em `agent/loop.py`:

1. envia a conversa e as definições das tools ao modelo;
2. se o modelo respondeu com pedidos de tool, executa e anexa os resultados à conversa;
3. repete até o modelo responder com texto.

A diferença para um pipeline fixo é que ninguém programou a sequência. Na pergunta sobre a margem da Nike, o modelo escolheu este caminho sozinho:

```
model   -> query_financials (receita e lucro bruto por trimestre)
model   -> calculator x7 (margem de cada trimestre, em paralelo)
model   -> search_filings (motivos da queda) + query_financials (conferir o Q4)
model   -> search_filings (explicação do trimestre mais recente)
model   -> calculator x3 (variação em pontos percentuais)
model   -> resposta final
```

Foram 6 chamadas ao modelo e 14 execuções de tool, em 30 segundos. Com `make ask` você vê esse caminho ao vivo, e ele fica gravado em `agent_runs` e `agent_steps`.

O preço dessa autonomia é imprevisibilidade: o agente pode dar voltas demais ou escolher mal. Por isso o loop tem um limite de passos (`AGENT_MAX_STEPS`), e por isso tudo é registrado.

## Guardrails

Guardrail é qualquer limite imposto por fora do modelo, que vale mesmo se o modelo errar. O prompt pede bom comportamento; o guardrail garante. O projeto tem quatro.

**1. SQL em três camadas.** A tool `query_financials` executa SQL escrito pelo LLM:

- o validador (`agent/sql_guard.py`) só aceita um único `SELECT` sobre as tabelas permitidas, e devolve um erro que o modelo consegue corrigir;
- o login `agent_ro` só tem `SELECT` em `companies`, `financial_facts` e `financial_metrics`, com transação read-only por padrão;
- queries demoradas são canceladas por `statement_timeout`, e o resultado é limitado a 200 linhas.

A sessão consegue desligar o modo read-only, e por isso ele não é a única barreira: mesmo assim o `INSERT` falha por falta de privilégio. Os casos estão testados em `tests/test_schema_integration.py` e `tests/test_agent_tools.py`.

**2. Citações que não podem ser inventadas.** Cada trecho ou resultado de query recebe um id do código (`S1`, `D1`), não do modelo. O modelo cita os ids no texto, e a lista de fontes no fim é montada pelo código a partir do registro (`agent/sources.py`). Um id citado que nenhuma tool devolveu é detectado e sinalizado.

**3. "Não encontrei" como resposta válida.** O prompt de sistema diz que a memória do modelo não é fonte e que dizer que não achou é melhor que um palpite. Perguntado sobre a receita da Amazon, que está fora da cobertura, o agente respondeu que não cobre a empresa e listou as cinco que cobre, sem inventar número.

**4. Limites do loop.** Máximo de passos por pergunta e máximo de tokens por resposta.

O que ainda não há: uma verificação automática de que cada afirmação bate com o trecho citado. É o que os evals medem por amostragem.

## Evals

Eval é um teste para um sistema cuja saída não é determinística: um conjunto fixo de perguntas, uma forma de rodar o sistema em cada uma, e uma forma de dar nota à saída. Sem isso, "troquei o prompt e ficou melhor" é impressão. Com isso, é uma medida que dá para repetir.

O conjunto fica em `evals/dataset.yaml`: 22 perguntas de quatro tipos.

| Tipo | Quantas | Exemplo | O que testa |
|---|---|---|---|
| Números | 8 | Qual o CAGR do lucro líquido da Coca-Cola entre 2020 e 2025? | SQL e calculadora |
| Texto | 9 | What drove the growth in Apple's Services net sales? | Busca nos relatórios |
| Misto | 1 | Margem da Nike no trimestre e a explicação da administração | As duas coisas juntas |
| Sem resposta | 4 | Qual foi a receita da Amazon em 2025? | Dizer "não encontrei" |

Todo gabarito foi lido dos dados ingeridos (a tabela `financial_metrics` e os trechos indexados), não escrito de memória. As perguntas sem resposta são tão importantes quanto as outras: um agente que nunca diz "não sei" acerta todas as perguntas respondíveis e ainda assim é perigoso.

O projeto mede duas coisas separadamente, porque falham por motivos diferentes.

**1. Recuperação (`make eval-retrieval`, gratuito).** Para as 10 perguntas respondidas com texto: em que posição aparece o primeiro trecho relevante? Não envolve LLM. As métricas:

- **hit@k:** em quantas perguntas o trecho certo ficou entre os k primeiros.
- **MRR (mean reciprocal rank):** média de 1 dividido pela posição do trecho certo. Vale 1,0 se ele é sempre o primeiro e 0,5 se é sempre o segundo.

| Configuração | hit@1 | hit@3 | hit@10 | MRR |
|---|---|---|---|---|
| Só vetor | 30% | 80% | 100% | 0,58 |
| Só palavra-chave | 30% | 60% | 100% | 0,52 |
| Híbrida (RRF) | 60% | 100% | 100% | 0,80 |
| Híbrida + re-ranker | 70% | 90% | 100% | 0,82 |

A busca híbrida põe o trecho certo no top 3 em todas as perguntas, o que nenhuma das duas buscas faz sozinha. O re-ranker ganha uma pergunta na primeira posição e perde uma do top 3, então não justifica os 0,7s a mais por busca.

**2. Resposta (`make eval`, pago).** O agente inteiro responde cada pergunta, e a resposta é avaliada de duas formas:

- **Checagens por código,** lidas do próprio run: citou alguma fonte? Todos os ids citados existem? Citou o relatório esperado? Quais tools usou?
- **Juiz LLM** (`evals/judge.py`): um segundo modelo lê a pergunta, os fatos esperados e a resposta, e devolve um veredito estruturado (correta, parcialmente correta, incorreta). É necessário porque a resposta é texto livre em dois idiomas: "\$478.8 billion" e "US\$ 478,8 bilhões" são o mesmo número, e uma comparação de strings daria nota à redação, não ao conteúdo.

Três cuidados com o juiz:

- **É um modelo diferente do avaliado** (Sonnet julgando Opus), para um modelo não dar nota a si mesmo.
- **É testado antes de ser usado** (`python -m evals.run judge-check`): tem que aprovar uma resposta perfeita e reprovar uma resposta vazia, um "não encontrei" numa pergunta respondível, e uma resposta confiante à pergunta errada.
- **Falha de execução não é resposta errada:** se a API cair no meio, o caso entra como erro e fica fora das taxas.

**Resultado das respostas** (22 perguntas, uma execução cada, Opus 5.5 avaliado pelo Sonnet 5.5):

| Medida | Resultado |
|---|---|
| Corretas, entre as perguntas respondíveis | 17 de 18 |
| Disse "não disponível", entre as sem resposta | 4 de 4 |
| Citou ao menos uma fonte | 18 de 18 |
| Citou o relatório esperado (perguntas de texto) | 10 de 10 |
| Runs com id de fonte que nenhuma tool devolveu | 0 de 22 |
| Mediana por pergunta | 8,9s, 5 passos, US\$ 0,06 |
| Custo do eval inteiro | US\$ 1,34 |

A única reprovada ensina mais sobre evals do que sobre o agente. A pergunta era "qual é a participação da Microsoft na OpenAI", sem data. O gabarito dizia 25%, tirado do 10-K de junho de 2026. O agente respondeu 24%, tirado de um 10-Q três meses mais novo, e ainda informou os 25% com a data. O agente estava certo e o gabarito estava velho. A pergunta agora fixa a data, e na reexecução o agente acertou. Vale como regra: quando um eval reprova, olhe primeiro se a culpa é do gabarito.

**Usando o eval para calibrar.** As primeiras respostas eram longas demais para uma thread: mediana de 197 palavras, 271 nas perguntas de texto. O prompt de sistema foi alterado em dois passos, com o eval rodando depois de cada um.

| Prompt | Mediana de palavras | Perguntas de texto | Corretas |
|---|---|---|---|
| "Tão curta quanto a pergunta permitir" | 197 | 271 | 17 de 18 |
| Resposta direta, 3 a 5 bullets, oferecer detalhe em vez de incluir | 155 | 213 | 18 de 18 |
| O mesmo, mais "cerca de 120 palavras, 200 como teto" | 138 | 167 | 18 de 18 |

Descrever o formato de uma boa resposta ajudou; dar um alvo concreto de palavras ajudou mais. O parâmetro `effort` não foi usado para isso: ele muda quanto o modelo investiga, e quase não muda o tamanho do texto. Sem o eval, a segunda linha teria parecido suficiente.

**Limites destes números.** O eval de respostas está perto do teto: serve para pegar uma regressão, mas não para escolher entre duas configurações boas. Cada pergunta rodou uma vez só, então não há medida da variação entre execuções. As perguntas foram escritas por quem construiu o sistema. Dez perguntas de recuperação significam que cada uma move a taxa em 10 pontos. E as queries de busca foram escritas por quem já tinha lido os trechos-alvo, então estão mais próximas do texto do relatório do que as de um usuário real: as taxas absolutas são otimistas, e o que vale é a comparação entre configurações, feita em condições iguais.
