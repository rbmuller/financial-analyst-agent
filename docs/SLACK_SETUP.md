# Creating the Slack app

The bot runs in Socket Mode: it opens an outbound connection (a WebSocket) to Slack and receives events through it. It needs no public URL, and it works from your own machine.

You will finish with two tokens in `.env`:

| Variable | Starts with | What it is for |
|---|---|---|
| `SLACK_APP_TOKEN` | `xapp-` | Opening the Socket Mode connection |
| `SLACK_BOT_TOKEN` | `xoxb-` | Reading mentions and posting messages as the bot |

You need permission to install apps in the workspace.

## 1. Create the app from the manifest

1. Open <https://api.slack.com/apps> and click **Create New App** → **From a manifest**.
2. Choose your workspace.
3. Select the **YAML** tab, paste the manifest below and confirm.

```yaml
display_information:
  name: SEC Analyst
  description: Answers questions about US-listed companies from their SEC filings
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

The manifest already turns on Socket Mode, the scopes and the events. What each item does:

| Item | Type | Why |
|---|---|---|
| `app_mentions:read` | scope | Receive the messages that mention the bot |
| `chat:write` | scope | Post and edit replies |
| `im:history` | scope | Read direct messages sent to the bot |
| `app_mention` | event | Fires when someone writes `@SEC Analyst` in a channel |
| `message.im` | event | Fires when someone sends the bot a direct message |

The bot does not read channel history: it only receives messages that mention it, and direct messages.

## 2. Generate the Socket Mode token (`xapp-`)

1. In the sidebar, open **Basic Information**.
2. Under **App-Level Tokens**, click **Generate Token and Scopes**.
3. Give it a name (for example `socket`), add the `connections:write` scope and click **Generate**.
4. Copy the `xapp-...` token.

## 3. Install to the workspace and get the bot token (`xoxb-`)

1. In the sidebar, open **Install App** and click **Install to Workspace**.
2. Review the permissions and authorize.
3. Copy the **Bot User OAuth Token** (`xoxb-...`).

## 4. Configure `.env`

```
SLACK_APP_TOKEN=xapp-...
SLACK_BOT_TOKEN=xoxb-...
```

The tokens grant access to the workspace: keep them only in `.env`, which is outside git. If one leaks, revoke it on the same page where it was generated.

## 5. Run and test

```bash
make slack
```

Wait for the line `Connected. Mention the bot in a channel or send it a direct message.` (loading the embedding model takes a few seconds). Then:

1. In a channel, invite the bot: `/invite @SEC Analyst`.
2. Ask: `@SEC Analyst what was Apple's revenue in the last fiscal year?`
3. The bot replies in the thread with a "thinking" message, updates the status while it works, and replaces the message with the answer and its sources.
4. In the same thread, ask a follow-up: `@SEC Analyst and Microsoft's?` In a channel the bot only sees messages that mention it, so the follow-up needs the mention too.
5. Also try a direct message to the bot, with no mention.

Stop the bot with `Ctrl+C`. It only answers while `make slack` is running.

The bot replies in the language of the question. Its fixed status and error messages ("Pensando…", "Fontes") are in Portuguese; they are constants in `slack_app/formatting.py`.

## Common problems

| Symptom | Likely cause |
|---|---|
| `SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set` | The tokens are missing from `.env` |
| `invalid_auth` on startup | The tokens are swapped (`xapp-` in place of `xoxb-`) or revoked |
| The bot does not react to a mention in a channel | The bot was not invited to the channel |
| Direct messages do not work | In **App Home**, the **Messages** tab must be enabled; reinstall the app if scopes changed |
| `missing_scope` in the logs | The app was changed after installation: reinstall under **Install App** |
