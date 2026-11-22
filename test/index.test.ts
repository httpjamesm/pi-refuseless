import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { getSupportedThinkingLevels, type RefreshModelsContext } from "@earendil-works/pi-ai/models";
import extension, {
	createRefuselessProvider,
	parseRefuselessModels,
	REFUSELESS_DEFAULT_BASE_URL,
	REFUSELESS_PROVIDER_ID,
	REFUSELESS_THINKING_LEVEL_MAP,
} from "../extensions/refuseless/index.ts";

describe("parseRefuselessModels", () => {
	it("parses minimal OpenAI /v1/models payloads with sensible defaults", () => {
		const models = parseRefuselessModels({
			object: "list",
			data: [
				{ id: "qwen3-235b-abliterated", object: "model", owned_by: "refuseless" },
				{ id: "llama-3.3-70b-abliterated", object: "model", owned_by: "refuseless" },
			],
		});

		assert.equal(models.length, 2);

		assert.deepEqual(models[0], {
			id: "qwen3-235b-abliterated",
			name: "qwen3-235b-abliterated",
			api: "openai-completions",
			provider: REFUSELESS_PROVIDER_ID,
			baseUrl: REFUSELESS_DEFAULT_BASE_URL,
			reasoning: true,
			thinkingLevelMap: REFUSELESS_THINKING_LEVEL_MAP,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 128_000,
			maxTokens: 16_384,
		});
		assert.deepEqual(getSupportedThinkingLevels(models[0]), ["off", "low", "medium", "high", "xhigh", "max"]);

		assert.equal(models[1].id, "llama-3.3-70b-abliterated");
		assert.equal(models[1].reasoning, false);
		assert.equal(models[1].thinkingLevelMap, undefined);
		assert.deepEqual(getSupportedThinkingLevels(models[1]), ["off"]);
	});

	it("extracts rich metadata (display name, context length, reasoning, vision, pricing) and filters non-chat models", () => {
		const models = parseRefuselessModels(
			{
				data: [
					{
						id: "deepseek-r1-abliterated",
						display_name: "DeepSeek R1 Abliterated",
						context_length: 163_840,
						max_output_tokens: 32_768,
						reasoning: true,
						input_modalities: ["text", "image"],
						pricing: { input_per_million: 0.8, output_per_million: 2.4 },
					},
					{
						id: "bge-m3-embedding",
						object: "model",
					},
					{
						id: "deepseek-r1-abliterated", // duplicate id
					},
				],
			},
			"https://custom.refuseless.example/v1/",
		);

		assert.equal(models.length, 1);
		assert.deepEqual(models[0], {
			id: "deepseek-r1-abliterated",
			name: "DeepSeek R1 Abliterated",
			api: "openai-completions",
			provider: REFUSELESS_PROVIDER_ID,
			baseUrl: "https://custom.refuseless.example/v1",
			reasoning: true,
			thinkingLevelMap: REFUSELESS_THINKING_LEVEL_MAP,
			input: ["text", "image"],
			cost: { input: 0.8, output: 2.4, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 163_840,
			maxTokens: 32_768,
		});
	});

	it("throws on invalid payload shape", () => {
		assert.throws(() => parseRefuselessModels({ invalid: true }), /Unexpected \/models response/);
	});
});

describe("createRefuselessProvider", () => {
	it("resolves REFUSELESS_API_KEY from environment and prefers stored credentials", async () => {
		const provider = createRefuselessProvider();
		const apiKeyAuth = provider.auth.apiKey;
		assert.ok(apiKeyAuth);

		const signal = new AbortController().signal;
		const envResult = await apiKeyAuth.resolve({
			ctx: {
				env: async (name) => (name === "REFUSELESS_API_KEY" ? "rf-env-key" : undefined),
				fileExists: async () => false,
			},
			signal,
		});
		assert.deepEqual(envResult, {
			auth: { apiKey: "rf-env-key" },
			source: "REFUSELESS_API_KEY",
		});

		const storedResult = await apiKeyAuth.resolve({
			ctx: {
				env: async () => "rf-env-key",
				fileExists: async () => false,
			},
			credential: { type: "api_key", key: "rf-stored-key" },
			signal,
		});
		assert.deepEqual(storedResult, {
			auth: { apiKey: "rf-stored-key" },
			env: undefined,
			source: "stored credential",
		});
	});

	it("fetches models dynamically, sends Bearer token, and publishes persisted catalog", async () => {
		const requests: { url: string; headers: Record<string, string> }[] = [];
		const mockFetch: typeof fetch = async (input, init) => {
			requests.push({
				url: String(input),
				headers: (init?.headers as Record<string, string>) ?? {},
			});
			return new Response(
				JSON.stringify({
					object: "list",
					data: [{ id: "kimi-k2-thinking-abliterated", context_length: 256_000 }],
				}),
				{ status: 200, headers: { "Content-Type": "application/json" } },
			);
		};

		const provider = createRefuselessProvider({ fetch: mockFetch });
		assert.equal(provider.getModels().length, 0);

		let persistedCount = 0;
		const context: RefreshModelsContext = {
			credential: { type: "api_key", key: "rf-secret-test" },
			allowNetwork: true,
			signal: new AbortController().signal,
			publish: async (publication) => {
				if (publication.persist) {
					persistedCount = publication.persist.models.length;
				}
				publication.update?.();
				return true;
			},
		};

		await provider.refreshModels?.(context);

		assert.equal(requests.length, 1);
		assert.equal(requests[0].url, "https://api.refuseless.com/v1/models");
		assert.equal(requests[0].headers.Authorization, "Bearer rf-secret-test");
		assert.equal(persistedCount, 1);
		assert.equal(provider.getModels().length, 1);
		assert.equal(provider.getModels()[0].id, "kimi-k2-thinking-abliterated");
		assert.equal(provider.getModels()[0].reasoning, true);
		assert.equal(provider.getModels()[0].contextWindow, 256_000);
	});

	it("restores stored models offline without making network calls", async () => {
		let fetchCalled = false;
		const mockFetch: typeof fetch = async () => {
			fetchCalled = true;
			throw new Error("should not be called");
		};

		const provider = createRefuselessProvider({ fetch: mockFetch });
		const storedModel = parseRefuselessModels({ data: [{ id: "cached-model" }] })[0];

		await provider.refreshModels?.({
			allowNetwork: false,
			stored: { models: [storedModel], checkedAt: Date.now() },
			signal: new AbortController().signal,
			publish: async (publication) => {
				publication.update?.();
				return true;
			},
		});

		assert.equal(fetchCalled, false);
		assert.deepEqual(
			provider.getModels().map((m) => m.id),
			["cached-model"],
		);
	});
});

describe("extension entrypoint", () => {
	it("registers the refuseless provider and refresh command", () => {
		const registeredProviders: unknown[] = [];
		const registeredCommands: string[] = [];

		extension({
			registerProvider: (provider: unknown) => registeredProviders.push(provider),
			registerCommand: (name: string) => registeredCommands.push(name),
		} as never);

		assert.equal(registeredProviders.length, 1);
		assert.deepEqual(registeredCommands, ["refuseless-refresh"]);
	});
});
