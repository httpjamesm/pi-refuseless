# pi-refuseless

A [Pi](https://pi.dev) provider extension that connects Pi to [Refuseless](https://refuseless.com).

Models are discovered dynamically from `GET /v1/models`, so new Refuseless models
show up without an extension update.

## Install

```bash
pi install git:github.com/httpjamesm/pi-refuseless
```

Pi clones the repository over HTTPS. The repository is private, so make sure git can
authenticate. If `git clone` asks for a username, route GitHub over SSH instead:

```bash
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

Then authenticate and pick a model:

```text
/login refuseless
/model
```

`/reload` picks up changes if you already had Pi running.

## Alternative: environment variable

```bash
export REFUSELESS_API_KEY=...
```

The env var is used when no key has been stored via `/login`.

## Commands

| Command | Description |
|---|---|
| `/login refuseless` | Store an API key in `~/.pi/agent/auth.json` |
| `/refuseless-refresh` | Force-refresh the model catalog from `/v1/models` |

## Requirements

- Pi 1.6.0 or newer.
- Node.js 22.18+ if you want to run the tests (`npm test`).

## How it works

`extensions/refuseless/index.ts` registers a provider backed by Pi's built-in
OpenAI Chat Completions streaming implementation. The provider declares a
`fetchModels` hook that maps the Refuseless `/v1/models` payload onto Pi's model
metadata (context window, max tokens, reasoning support, modalities, cost).

## Development

```bash
npm test
```

To iterate on changes, edit this checkout and install it in place instead of the git
copy:

```bash
pi remove git:github.com/httpjamesm/pi-refuseless
pi install ./pi-refuseless
```

Or try a one-off session without touching settings:

```bash
pi -e ./pi-refuseless
```

Changes pushed to the git copy need `pi update --extensions` on every machine that
installed it that way.

## License

MIT
