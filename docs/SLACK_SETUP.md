# Criando o app no Slack

O bot roda em Socket Mode: ele abre uma conexão de saída (WebSocket) com o Slack e recebe os eventos por ela. Não precisa de URL pública, e funciona na sua máquina.

Você vai sair daqui com dois tokens no `.env`:

| Variável | Começa com | Para que serve |
|---|---|---|
| `SLACK_APP_TOKEN` | `xapp-` | Abrir a conexão do Socket Mode |
| `SLACK_BOT_TOKEN` | `xoxb-` | Ler menções e postar mensagens como o bot |

É preciso ter permissão para instalar apps no workspace.

## 1. Criar o app a partir do manifesto

1. Abra <https://api.slack.com/apps> e clique em **Create New App** → **From a manifest**.
2. Escolha o workspace da TBDC.
3. Selecione a aba **YAML**, cole o manifesto abaixo e confirme.

```yaml
display_information:
  name: SEC Analyst
  description: Responde perguntas sobre empresas listadas nos EUA a partir dos relatórios da SEC
features:
  app_home:
    messages_tab_enabled: true
    messages_tab_read_only_enabled: false
  bot_user:
    display_name: SEC Analyst
    always_online: true
oauth_config:
  scopes:
    bot:
      - app_mentions:read
      - chat:write
      - im:history
settings:
  event_subscriptions:
    bot_events:
      - app_mention
      - message.im
  interactivity:
    is_enabled: false
  org_deploy_enabled: false
  socket_mode_enabled: true
  token_rotation_enabled: false
```

O manifesto já liga o Socket Mode, os scopes e os eventos. O que cada item faz:

| Item | Tipo | Por quê |
|---|---|---|
| `app_mentions:read` | scope | Receber as mensagens em que o bot é mencionado |
| `chat:write` | scope | Postar e editar as respostas |
| `im:history` | scope | Ler mensagens diretas enviadas ao bot |
| `app_mention` | evento | Dispara quando alguém escreve `@SEC Analyst` num canal |
| `message.im` | evento | Dispara quando alguém manda mensagem direta ao bot |

O bot não lê o histórico dos canais: só recebe as mensagens que o mencionam e as mensagens diretas.

## 2. Gerar o token do Socket Mode (`xapp-`)

1. No menu lateral, abra **Basic Information**.
2. Em **App-Level Tokens**, clique em **Generate Token and Scopes**.
3. Dê um nome (por exemplo `socket`), adicione o scope `connections:write` e clique em **Generate**.
4. Copie o token `xapp-...`.

## 3. Instalar no workspace e pegar o token do bot (`xoxb-`)

1. No menu lateral, abra **Install App** e clique em **Install to Workspace**.
2. Revise as permissões e autorize.
3. Copie o **Bot User OAuth Token** (`xoxb-...`).

## 4. Configurar o `.env`

```
SLACK_APP_TOKEN=xapp-...
SLACK_BOT_TOKEN=xoxb-...
```

Os tokens dão acesso ao workspace: ficam só no `.env`, que está fora do git. Se um deles vazar, revogue-o na mesma página em que foi gerado.

## 5. Rodar e testar

```bash
make slack
```

Espere a linha `Connected. Mention the bot in a channel or send it a direct message.` (o carregamento do modelo de embedding leva alguns segundos). Depois:

1. Num canal, convide o bot: `/invite @SEC Analyst`.
2. Pergunte: `@SEC Analyst qual foi a receita da Apple no último ano fiscal?`
3. O bot responde na thread com "Pensando…", atualiza o status enquanto trabalha e troca a mensagem pela resposta com as fontes.
4. Na mesma thread, faça uma continuação: `@SEC Analyst e a da Microsoft?`
5. Teste também uma mensagem direta ao bot, sem menção.

Pare o bot com `Ctrl+C`. Ele só responde enquanto o `make slack` estiver rodando.

## Problemas comuns

| Sintoma | Causa provável |
|---|---|
| `SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set` | Faltam os tokens no `.env` |
| `invalid_auth` ao iniciar | Token trocado de lugar (`xapp-` no lugar do `xoxb-`) ou revogado |
| Bot não reage à menção no canal | Bot não foi convidado para o canal |
| Mensagem direta não funciona | Em **App Home**, a aba **Messages** precisa estar habilitada; reinstale o app se mudou scopes |
| `missing_scope` nos logs | O app foi alterado depois de instalado: reinstale em **Install App** |
