# Concepts

The concepts this project is built on, each explained with an example taken from the project itself.

The search examples use the question `Why did gross margin decline?`, filtered to Nike, fiscal year 2026. To reproduce them:

```bash
make search Q="Why did gross margin decline?" ARGS="--ticker NKE --year 2026 --top-k 4 --mode vector"
```

## Embedding

An embedding is a list of numbers that represents the meaning of a text. Texts with similar meaning get similar lists, even when they share no words. The model used here (`BAAI/bge-small-en-v1.5`, running locally) returns 384 numbers for any text.

Inside, the model splits the text into tokens (word pieces, from a fixed vocabulary of about 30,000), gives each token a starting vector, and passes everything through layers that mix the tokens with one another. That mixing is why "margin" in "gross margin declined" and in "margin of error" end up with different representations. At the end, the whole text is summarized into a single vector.

The model only converts text into a vector. Storing the vector is our code's job (`rag/indexing.py`): it goes into the same row of the `chunks` table that the text came from.

Two details of `rag/embeddings/local.py`:

- **Queries and documents are embedded differently.** BGE models were trained with a fixed instruction at the start of the query (`Represent this sentence for searching relevant passages: `) and nothing at the start of the document. That is why the interface has both `embed_query` and `embed_documents`.
- **Vectors are normalized** (length 1), which keeps cosine comparison stable.

The `EmbeddingProvider` interface (`rag/embeddings/base.py`) isolates the rest of the code from the model: switching to a paid provider means writing one new class.

## Vector database

A vector database stores embeddings and answers the question "which vectors are closest to this one?". Here it is Postgres itself, with the pgvector extension: the `chunks` table has an `embedding vector(384)` column next to the text and its metadata.

Keeping everything in one database has a practical advantage: the same query filters by `ticker = 'NKE'` and orders by semantic closeness, with no second system to keep in sync.

Closeness is measured by cosine distance, which compares the direction of two vectors and ignores their length. In pgvector it is the `<=>` operator:

```sql
SELECT '[1,0]'::vector <=> '[0,1]'::vector;  -- 1 (orthogonal directions, nothing in common)
SELECT '[1,0]'::vector <=> '[2,0]'::vector;  -- 0 (same direction, same meaning)
```

## HNSW index

Without an index, finding the nearest neighbours means computing the distance to every chunk. HNSW (Hierarchical Navigable Small World) avoids that with a layered graph: the upper layers have few points and long links, the lower layers have every point and short links. A search enters at the top, jumps to the right region and refines as it descends.

The result is approximate: the index may miss a true neighbour in exchange for speed. For the 7,500 chunks in this project an exact search would also be fast; the index is here because it is what you use once the volume grows.

```sql
CREATE INDEX chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
```

`vector_cosine_ops` has to match the operator used in the query (`<=>`); with a different operator Postgres ignores the index.

**The trap with filters.** The index returns the nearest neighbours in the whole table, and `WHERE ticker = 'NKE'` is applied afterwards. If the 100 nearest neighbours all belong to another company, the filtered search comes back empty even though Nike chunks exist. The fix is pgvector's *iterative scan* (version 0.8 onwards), which keeps walking the graph until enough rows pass the filter. It is enabled in `rag/search.py` (`SET LOCAL hnsw.iterative_scan`) and covered by the test `test_filters_are_applied_inside_the_vector_scan`, which builds exactly that scenario.

## Chunking

Chunking is splitting each document into smaller pieces before embedding them. There are two reasons:

- the model reads at most 512 tokens, and the MD&A of a Nike 10-K is about 70,000 characters;
- a single vector for a long text is too vague to match a specific question.

The strategy in `rag/chunking.py`:

1. The filing was already split by section at ingestion (Item 7, Item 1A and so on), so no chunk mixes sections.
2. Inside a section, whole paragraphs are packed together up to 400 tokens. A paragraph longer than that is cut by sentence.
3. Neighbouring chunks share up to 60 tokens (*overlap*), so an idea cut at a boundary appears whole in at least one of them.

The 60 filings became 7,542 chunks, averaging 330 tokens.

**Context header.** A chunk like "Gross margin contraction of approximately 200 basis points" does not say whose margin or when. Before embedding, the code prefixes one line of context:

```
NIKE, Inc. (NKE) 10-Q FY2026 Q3 | Part I, Item 2: Management's Discussion and Analysis
Reported EBIT decreased 4% reflecting higher reported revenues more than offset by: ...
```

The header goes into the vector; the `content` column keeps only the original text, which is what citations show.

## Hybrid search

Two searches with complementary strengths, merged into one result.

- **Vector search** finds passages that mean the same as the question, even in other words. The question says "decline", and the top vector result says "Gross margin contraction of approximately 700 basis points".
- **Keyword search** (Postgres full-text, the `content_tsv` column with a GIN index) finds passages with the exact terms. It is better for product names, acronyms and numbers, which embeddings tend to blur.

The question's terms are combined with OR, not AND: a question almost never appears word for word in a filing, and the ranking already favours passages that cover more of its terms.

**Reciprocal Rank Fusion (RRF).** The two searches return scores on different scales (cosine distance and full-text rank) that cannot be added. RRF ignores the scores and uses only the position: each passage earns `1 / (60 + position)` in each list, and the parts are summed.

In the example, the top two hybrid results were nearly tied:

```
1st: vector #12, keyword #1   ->  1/(60+12) + 1/(60+1)  = 0.0139 + 0.0164 = 0.0303
2nd: vector #1,  keyword #13  ->  1/(60+1)  + 1/(60+13) = 0.0164 + 0.0137 = 0.0301
```

The first is the 10-Q's gross margin table followed by its explanation ("Consolidated gross margin was 300 basis points lower than the prior year primarily due to..."), which contains the question's exact terms. The second is the best semantic match. Hybrid search brings both to the top; either search alone would leave one of them in 12th or 13th place.

## Re-ranking

Re-ranking is a second stage that reorders the search candidates with a more precise, slower model.

The embedding model is a *bi-encoder*: it produces the query vector and each passage vector separately, never seeing the two together. That is what allows passage vectors to be computed once, at indexing time. The re-ranker is a *cross-encoder*: it reads the query and the passage together and returns a relevance score. It is more precise, but it has to run once per candidate on every question, so it is applied only to the 40 or so passages hybrid search already selected.

In the example, with `--rerank` the top spot went to a passage that was 7th in vector search and 22nd in keyword search, and that states the cause directly:

> Gross margin contraction of 330 basis points, primarily due to new tariffs, partially offset by lower warehousing and logistics costs due to channel mix, and lower inventory obsolescence reserves.

Not everything improves: third place went to a generic Risk Factors passage, which discusses margins in principle and not what happened.

The cost: the search went from 0.04 s to 0.7 s on CPU. The re-ranker used (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is small and was trained on web search, not financial text; whether it pays off here is a question for the evals (below), which is why it is optional.

## LLM

An LLM (large language model) is a model that takes text in and generates text. Here it is Claude, accessed through the Anthropic API. Three properties shape the whole project:

- **It has no memory between calls.** Every request carries the entire conversation again. The agent's "state" is the `messages` list in `agent/loop.py`.
- **It knows only what it saw in training and what is in the request.** It does not know the 10-Q Nike filed last month unless that text is put into the conversation.
- **It generates plausible text, not verified text.** Asked for a number it does not have, it may produce one that looks right. This is called hallucination, and it is what RAG, tools and guardrails exist to contain.

Billing is per token (word piece), with different prices for input and output. The example question about Nike's margin used 65,000 input tokens and 3,200 output tokens, and cost \$0.17.

**Prompt caching.** On every turn of the loop the whole conversation is sent again, so the same tokens would be paid for several times. With caching, the leading part that has not changed (tool definitions, system prompt, earlier turns) costs a fraction of the price. The cache compares the start of the request byte for byte, which is why the system prompt in `agent/prompts.py` contains no date and nothing else that varies.

## RAG

RAG (retrieval-augmented generation) means fetching the relevant passages first and handing them to the LLM together with the question, so that it answers from them and not from memory.

In this project it is the earlier pieces working together:

1. **Indexing (beforehand, once):** filings → sections → chunks → embeddings in Postgres.
2. **Retrieval (on every question):** hybrid search returns the 6 to 8 most relevant chunks.
3. **Generation:** Claude receives those chunks and writes the answer, citing which chunk each statement came from.

RAG addresses two LLM limitations at once: it gives the model access to documents it has never seen, and it makes the answer checkable, because each statement points at a passage that exists.

## Tool use

Tool use is the mechanism by which the LLM asks our code to do something. The model never executes anything: it returns a structured request, and the program decides whether and how to carry it out.

Every tool has two faces (see `agent/tools/base.py`):

- **To the model:** a name, a description and a JSON Schema for the arguments. That text is all the model knows about the tool, so the description says when to use it, not only what it does.
- **To the code:** an ordinary Python function.

In the example question, the model returned this request instead of text:

```json
{"type": "tool_use", "name": "calculator",
 "input": {"operation": "ratio_percent", "values": [4530000000, 11279000000]}}
```

The code did the arithmetic and returned `{"result": 40.1631}` in a `tool_result` message. Only then did the model write "40.2%".

The project's three tools:

| Tool | What for | Why it exists |
|---|---|---|
| `search_filings` | Passages from the filings (the RAG part) | What management wrote is not in any table |
| `query_financials` | SQL over the XBRL figures | Semantic search is poor at exact numbers; SQL is exact |
| `calculator` | Percent change, CAGR, average, ratio | An LLM predicts plausible digits; it does not compute |

## Agent

An agent is an LLM inside a loop in which the model itself decides the next step. The whole loop is in `agent/loop.py`:

1. send the conversation and the tool definitions to the model;
2. if the model answered with tool requests, run them and append the results to the conversation;
3. repeat until the model answers with text.

The difference from a fixed pipeline is that nobody programmed the sequence. For the Nike margin question, the model chose this path on its own:

```
model   -> query_financials (revenue and gross profit by quarter)
model   -> calculator x7 (margin of each quarter, in parallel)
model   -> search_filings (reasons for the decline) + query_financials (cross-check Q4)
model   -> search_filings (explanation for the latest quarter)
model   -> calculator x3 (change in percentage points)
model   -> final answer
```

That was 6 model calls and 14 tool executions in 30 seconds. `make ask` shows this path live, and it is recorded in `agent_runs` and `agent_steps`.

The price of that autonomy is unpredictability: the agent may take too many turns or choose badly. That is why the loop has a step limit (`AGENT_MAX_STEPS`), and why everything is recorded.

## Guardrails

A guardrail is any limit imposed from outside the model, one that holds even when the model gets it wrong. The prompt asks for good behaviour; the guardrail enforces it. The project has four.

**1. Three layers around SQL.** The `query_financials` tool runs SQL written by the LLM:

- the validator (`agent/sql_guard.py`) accepts only a single `SELECT` over the allowed tables, and returns an error the model can act on;
- the `agent_ro` login can only `SELECT` from `companies`, `financial_facts` and `financial_metrics`, in transactions that are read-only by default;
- slow queries are cancelled by `statement_timeout`, and results are capped at 200 rows.

A session can switch read-only mode off, which is why it is not the only barrier: the `INSERT` still fails for lack of privilege. These cases are tested in `tests/test_schema_integration.py` and `tests/test_agent_tools.py`.

**2. Citations that cannot be invented.** Every passage or query result gets an id from the code (`S1`, `D1`), not from the model. The model cites ids in the text, and the source list at the end is built by code from the registry (`agent/sources.py`). A cited id that no tool returned is detected and flagged.

**3. "Not found" as a valid answer.** The system prompt says the model's memory is not a source, and that saying it found nothing is better than a guess. Asked for Amazon's revenue, which is outside the coverage, the agent replied that it does not cover the company and listed the five it does, without inventing a number.

**4. Loop limits.** A maximum number of steps per question and a maximum number of tokens per answer.

What is still missing: an automatic check that each statement matches the passage it cites. The evals measure that only by sampling.

## Evals

An eval is a test for a system whose output is not deterministic: a fixed set of questions, a way to run the system on each one, and a way to grade the output. Without it, "I changed the prompt and it got better" is an impression. With it, it is a measurement you can repeat.

The set is in `evals/dataset.yaml`: 22 questions of four kinds.

| Kind | Count | Example | What it tests |
|---|---|---|---|
| Figures | 8 | What was Coca-Cola's net income CAGR between 2020 and 2025? | SQL and the calculator |
| Text | 9 | What drove the growth in Apple's Services net sales? | Search over the filings |
| Mixed | 1 | Nike's margin in a quarter and management's explanation | Both together |
| Unanswerable | 4 | What was Amazon's revenue in 2025? | Saying "not found" |

Every expected answer was read from the ingested data (the `financial_metrics` table and the indexed passages), not written from memory. The unanswerable questions matter as much as the others: an agent that never says "I don't know" gets every answerable question right and is still dangerous.

The project measures two things separately, because they fail for different reasons.

**1. Retrieval (`make eval-retrieval`, free).** For the 10 questions answered from text: at what position does the first relevant passage appear? No LLM is involved. The metrics:

- **hit@k:** on how many questions the right passage was among the first k.
- **MRR (mean reciprocal rank):** the average of 1 divided by the position of the right passage. It is 1.0 if the passage is always first and 0.5 if it is always second.

| Setup | hit@1 | hit@3 | hit@10 | MRR |
|---|---|---|---|---|
| Vector only | 30% | 80% | 100% | 0.58 |
| Keyword only | 30% | 60% | 100% | 0.52 |
| Hybrid (RRF) | 60% | 100% | 100% | 0.80 |
| Hybrid + re-ranker | 70% | 90% | 100% | 0.82 |

Hybrid search puts the right passage in the top 3 on every question, which neither search does alone. The re-ranker gains one question at first place and loses one from the top 3, so it does not justify the extra 0.7 s per search.

**2. Answers (`make eval`, paid).** The whole agent answers each question, and the answer is graded in two ways:

- **Checks in code,** read from the run itself: did it cite a source? Do all cited ids exist? Did it cite the expected filing? Which tools did it use?
- **LLM judge** (`evals/judge.py`): a second model reads the question, the expected facts and the answer, and returns a structured verdict (correct, partially correct, incorrect). It is needed because the answer is free text in two languages: "\$478.8 billion" and "US\$ 478,8 bilhões" are the same number, and a string comparison would grade the wording, not the content.

Three precautions with the judge:

- **It is a different model from the one being graded** (Sonnet judging Opus), so a model does not grade itself.
- **It is tested before it is used** (`python -m evals.run judge-check`): it has to pass a perfect answer and fail an empty answer, a "not found" on an answerable question, and a confident answer to the wrong question.
- **A failed run is not a wrong answer:** if the API drops mid-run, the case is recorded as an error and left out of the rates.

**Answer results** (22 questions, one run each, Opus 5.5 graded by Sonnet 5.5):

| Measure | Result |
|---|---|
| Correct, among answerable questions | 17 of 18 |
| Said "not available", among unanswerable questions | 4 of 4 |
| Cited at least one source | 18 of 18 |
| Cited the expected filing (text questions) | 10 of 10 |
| Runs citing a source id no tool returned | 0 of 22 |
| Median per question | 8.9 s, 5 steps, \$0.06 |
| Cost of the whole eval | \$1.34 |

The single miss says more about evals than about the agent. The question was "what is Microsoft's stake in OpenAI", with no date. The expected answer said 25%, taken from the June 2026 10-K. The agent answered 24%, from a 10-Q three months newer, and also gave the 25% figure with its date. The agent was right and the expected answer was stale. The question now pins the date, and on a re-run the agent got it right. A rule worth keeping: when an eval fails a case, check first whether the expected answer is at fault.

**Using the eval to calibrate.** The first answers were too long for a chat thread: a median of 197 words, 271 on text questions. The system prompt was changed in two steps, with the eval run after each.

| Prompt | Median words | Text questions | Correct |
|---|---|---|---|
| "As short as the question allows" | 197 | 271 | 17 of 18 |
| Direct answer, 3 to 5 bullets, offer detail instead of including it | 155 | 213 | 18 of 18 |
| Same, plus "about 120 words, 200 as the ceiling" | 138 | 167 | 18 of 18 |

Describing the shape of a good answer helped; giving a concrete word target helped more. The `effort` parameter was not used for this: it changes how much the model investigates, and barely changes the length of the text. Without the eval, the second row would have looked good enough.

**Limits of these numbers.** The answer eval is close to its ceiling: it can catch a regression, but it cannot choose between two good configurations. Each question ran only once, so there is no measure of run-to-run variation. The questions were written by the person who built the system. Ten retrieval questions means each one moves a rate by 10 points. And the search queries were written by someone who had already read the target passages, so they are closer to the filing's wording than a real user's would be: the absolute rates are optimistic, and what holds is the comparison between setups, made on equal terms.
