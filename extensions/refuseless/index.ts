/**
 * Refuseless Provider Extension
 *
 * Connects Pi to Refuseless (https://api.refuseless.com/v1) using Pi's built-in
 * OpenAI Chat Completions streaming implementation and dynamic model discovery
 * via GET /v1/models.
 *
 * Authentication:
 *   - Set REFUSELESS_API_KEY in your environment, or
 *   - Run `/login refuseless` to store an API key in ~/.pi/agent/auth.json
 *
 * Usage:
 *   Auto-discovered from ~/.pi/agent/extensions/refuseless/
 *   Run `/reload` after edits, `/refuseless-refresh` to force-refresh models,
 *   and `/model` to select a Refuseless model.
 */

import type { ApiKeyAuth, Model, Provider, ThinkingLevelMap } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/compat";
import { createProvider, type RefreshModelsContext } from "@earendil-works/pi-ai/models";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const REFUSELESS_PROVIDER_ID = "refuseless";
export const REFUSELESS_PROVIDER_NAME = "Refuseless";
export const REFUSELESS_DEFAULT_BASE_URL = "https://api.refuseless.com/v1";
export const REFUSELESS_API_KEY_ENV_VARS = ["REFUSELESS_API_KEY"] as const;

const DEFAULT_CONTEXT_WINDOW = 128_000;
const DEFAULT_MAX_TOKENS = 16_384;

export const REFUSELESS_THINKING_LEVEL_MAP: ThinkingLevelMap = {
	off: "none",
	minimal: null,
	low: "low",
	medium: "medium",
	high: "high",
	xhigh: "xhigh",
	max: "max",
};

const NON_CHAT_MODEL_RE = /(?:^|[-_/:])(embed(?:ding)?|rerank(?:er)?|whisper|tts|moderation|transcrib)/i;
const REASONING_MODEL_ID_RE =
	/(?:^|[-_/:])(r1|qwq|thinking|reason(?:er|ing)?|deepseek-v3\.[12]|deepseek-v4(?:\.\d+)?|qwen3(?:\.\d+)?|glm-4\.[5-9]|glm-5(?:\.\d+)?|kimi-k[23](?:\.\d+)?|minimax-m[123](?:\.\d+)?|mimo-v2(?:\.\d+)?|step-3(?:\.\d+)?|gemma-4|gpt-oss)(?:$|[-_/:.])/i;

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asNonEmptyString(value: unknown): string | undefined {
	return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function asPositiveInteger(value: unknown): number | undefined {
	if (typeof value === "number" && Number.isFinite(value) && value > 0) {
		return Math.floor(value);
	}
	if (typeof value === "string" && value.trim().length > 0) {
		const parsed = Number(value);
		if (Number.isFinite(parsed) && parsed > 0) {
			return Math.floor(parsed);
		}
	}
	return undefined;
}

function asNonNegativeNumber(value: unknown): number | undefined {
	if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
		return value;
	}
	if (typeof value === "string" && value.trim().length > 0) {
		const parsed = Number(value);
		if (Number.isFinite(parsed) && parsed >= 0) {
			return parsed;
		}
	}
	return undefined;
}

function normalizeBaseUrl(baseUrl: string): string {
	return baseUrl.replace(/\/+$/, "");
}

function extractStringArray(value: unknown): string[] {
	if (!Array.isArray(value)) return [];
	return value
		.filter((entry): entry is string => typeof entry === "string")
		.map((entry) => entry.trim().toLowerCase());
}

function detectReasoning(entry: JsonRecord, id: string): boolean {
	if (typeof entry.reasoning === "boolean") return entry.reasoning;
	if (typeof entry.supports_reasoning === "boolean") return entry.supports_reasoning;
	if (typeof entry.thinking === "boolean") return entry.thinking;

	if (isRecord(entry.capabilities)) {
		if (typeof entry.capabilities.reasoning === "boolean") return entry.capabilities.reasoning;
		if (typeof entry.capabilities.thinking === "boolean") return entry.capabilities.thinking;
	} else {
		const caps = extractStringArray(entry.capabilities);
		if (caps.includes("reasoning") || caps.includes("thinking")) return true;
	}

	const supportedParams = extractStringArray(entry.supported_parameters);
	if (supportedParams.includes("reasoning") || supportedParams.includes("reasoning_effort")) {
		return true;
	}

	const tags = extractStringArray(entry.tags);
	if (tags.includes("reasoning") || tags.includes("thinking")) {
		return true;
	}

	return REASONING_MODEL_ID_RE.test(id);
}

function detectInputModalities(entry: JsonRecord): ("text" | "image")[] {
	const modalities = new Set<string>();

	for (const value of [
		entry.input,
		entry.input_modalities,
		entry.modalities,
		isRecord(entry.architecture) ? entry.architecture.input_modalities : undefined,
	]) {
		for (const item of extractStringArray(value)) {
			modalities.add(item);
		}
	}

	if (isRecord(entry.architecture) && typeof entry.architecture.modality === "string") {
		const archModality = entry.architecture.modality.toLowerCase();
		if (archModality.includes("image") || archModality.includes("vision")) {
			modalities.add("image");
		}
	}

	if (entry.vision === true || entry.supports_vision === true) {
		modalities.add("image");
	}
	if (isRecord(entry.capabilities) && entry.capabilities.vision === true) {
		modalities.add("image");
	} else if (extractStringArray(entry.capabilities).includes("vision")) {
		modalities.add("image");
	}

	return modalities.has("image") || modalities.has("vision") ? ["text", "image"] : ["text"];
}

function normalizeRatePerMillion(value: unknown, isPerToken: boolean): number {
	const num = asNonNegativeNumber(value);
	if (num === undefined) return 0;
	return isPerToken ? num * 1_000_000 : num;
}

function extractCost(entry: JsonRecord): Model<"openai-completions">["cost"] {
	const costRecord = isRecord(entry.cost) ? entry.cost : undefined;
	if (costRecord) {
		return {
			input: asNonNegativeNumber(costRecord.input) ?? 0,
			output: asNonNegativeNumber(costRecord.output) ?? 0,
			cacheRead: asNonNegativeNumber(costRecord.cacheRead ?? costRecord.cache_read) ?? 0,
			cacheWrite: asNonNegativeNumber(costRecord.cacheWrite ?? costRecord.cache_write) ?? 0,
		};
	}

	const pricing = isRecord(entry.pricing) ? entry.pricing : undefined;
	if (pricing) {
		const hasPerMillionKeys =
			pricing.input_per_million !== undefined ||
			pricing.output_per_million !== undefined ||
			pricing.prompt_per_million !== undefined ||
			pricing.completion_per_million !== undefined;

		if (hasPerMillionKeys) {
			return {
				input: asNonNegativeNumber(pricing.input_per_million ?? pricing.prompt_per_million) ?? 0,
				output: asNonNegativeNumber(pricing.output_per_million ?? pricing.completion_per_million) ?? 0,
				cacheRead: asNonNegativeNumber(pricing.cache_read_per_million) ?? 0,
				cacheWrite: asNonNegativeNumber(pricing.cache_write_per_million) ?? 0,
			};
		}

		// OpenRouter-style pricing uses per-token strings/numbers (typically < 0.01);
		// direct per-million pricing uses larger numbers. Convert per-token values safely.
		const rawInput = pricing.input ?? pricing.prompt;
		const rawOutput = pricing.output ?? pricing.completion;
		const parsedInput = asNonNegativeNumber(rawInput);
		const parsedOutput = asNonNegativeNumber(rawOutput);
		const looksPerToken =
			(typeof rawInput === "string" || typeof rawOutput === "string") &&
			((parsedInput !== undefined && parsedInput > 0 && parsedInput < 0.01) ||
				(parsedOutput !== undefined && parsedOutput > 0 && parsedOutput < 0.01));

		return {
			input: normalizeRatePerMillion(rawInput, looksPerToken),
			output: normalizeRatePerMillion(rawOutput, looksPerToken),
			cacheRead: normalizeRatePerMillion(pricing.cache_read ?? pricing.input_cache_read, looksPerToken),
			cacheWrite: normalizeRatePerMillion(pricing.cache_write ?? pricing.input_cache_write, looksPerToken),
		};
	}

	return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
}

function isChatModelEntry(entry: JsonRecord, id: string): boolean {
	const type = asNonEmptyString(entry.type)?.toLowerCase();
	if (type && type !== "chat" && type !== "model" && type !== "text" && type !== "llm") {
		return false;
	}
	return !NON_CHAT_MODEL_RE.test(id);
}

/**
 * Parse a GET /v1/models response payload from Refuseless into Pi chat models.
 */
export function parseRefuselessModels(
	payload: unknown,
	baseUrl: string = REFUSELESS_DEFAULT_BASE_URL,
): Model<"openai-completions">[] {
	const rawList = Array.isArray(payload)
		? payload
		: isRecord(payload) && Array.isArray(payload.data)
			? payload.data
			: null;

	if (!rawList) {
		throw new Error("Unexpected /models response format from Refuseless");
	}

	const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
	const seen = new Set<string>();
	const models: Model<"openai-completions">[] = [];

	for (const item of rawList) {
		if (!isRecord(item)) continue;
		const id = asNonEmptyString(item.id);
		if (!id || seen.has(id) || !isChatModelEntry(item, id)) continue;
		seen.add(id);

		const name = asNonEmptyString(item.name) ?? asNonEmptyString(item.display_name) ?? id;
		const contextWindow =
			asPositiveInteger(item.context_window) ??
			asPositiveInteger(item.context_length) ??
			asPositiveInteger(item.max_model_len) ??
			asPositiveInteger(item.max_context_length) ??
			DEFAULT_CONTEXT_WINDOW;

		const rawMaxTokens =
			asPositiveInteger(item.max_output_tokens) ??
			asPositiveInteger(item.max_completion_tokens) ??
			asPositiveInteger(item.max_tokens) ??
			DEFAULT_MAX_TOKENS;
		const maxTokens = Math.min(rawMaxTokens, contextWindow);

		const reasoning = detectReasoning(item, id);

		models.push({
			id,
			name,
			api: "openai-completions",
			provider: REFUSELESS_PROVIDER_ID,
			baseUrl: normalizedBaseUrl,
			reasoning,
			...(reasoning ? { thinkingLevelMap: { ...REFUSELESS_THINKING_LEVEL_MAP } } : {}),
			input: detectInputModalities(item),
			cost: extractCost(item),
			contextWindow,
			maxTokens,
		});
	}

	return models;
}

/**
 * Fetch the live model catalog from Refuseless GET /v1/models.
 */
export async function fetchRefuselessModels(
	context: RefreshModelsContext,
	fetchImpl: typeof fetch = fetch,
): Promise<Model<"openai-completions">[]> {
	const baseUrl = normalizeBaseUrl(context.baseUrl ?? REFUSELESS_DEFAULT_BASE_URL);
	const apiKey = context.credential?.type === "api_key" ? context.credential.key : undefined;

	const headers: Record<string, string> = {
		Accept: "application/json",
	};
	if (apiKey) {
		headers.Authorization = `Bearer ${apiKey}`;
	}

	const response = await fetchImpl(`${baseUrl}/models`, {
		method: "GET",
		headers,
		signal: context.signal,
	});

	if (!response.ok) {
		throw new Error(`Refuseless /models request failed with HTTP ${response.status} ${response.statusText}`.trim());
	}

	const payload = (await response.json()) as unknown;
	return parseRefuselessModels(payload, baseUrl);
}

function createRefuselessApiKeyAuth(): ApiKeyAuth {
	return {
		name: "Refuseless API key",
		login: async (interaction) => {
			interaction.signal.throwIfAborted();
			const key = await interaction.prompt({
				type: "secret",
				message: "Enter Refuseless API key",
			});
			interaction.signal.throwIfAborted();
			return { type: "api_key", key };
		},
		resolve: async ({ ctx, credential, signal }) => {
			signal.throwIfAborted();
			if (credential?.key) {
				return {
					auth: { apiKey: credential.key },
					env: credential.env,
					source: "stored credential",
				};
			}
			for (const envVar of REFUSELESS_API_KEY_ENV_VARS) {
				const value = await ctx.env(envVar);
				signal.throwIfAborted();
				if (value) {
					return { auth: { apiKey: value }, source: envVar };
				}
			}
			return undefined;
		},
	};
}

export interface CreateRefuselessProviderOptions {
	baseUrl?: string;
	fetch?: typeof fetch;
}

export function createRefuselessProvider(
	options: CreateRefuselessProviderOptions = {},
): Provider<"openai-completions"> {
	const baseUrl = normalizeBaseUrl(options.baseUrl ?? REFUSELESS_DEFAULT_BASE_URL);
	const fetchImpl = options.fetch ?? fetch;

	return createProvider({
		id: REFUSELESS_PROVIDER_ID,
		name: REFUSELESS_PROVIDER_NAME,
		baseUrl,
		auth: {
			apiKey: createRefuselessApiKeyAuth(),
		},
		models: [],
		api: openAICompletionsApi(),
		fetchModels: (context) =>
			fetchRefuselessModels(
				{
					...context,
					baseUrl: context.baseUrl ?? baseUrl,
				},
				fetchImpl,
			),
	});
}

export default function (pi: ExtensionAPI) {
	pi.registerProvider(createRefuselessProvider());

	pi.registerCommand("refuseless-refresh", {
		description: "Force-refresh the Refuseless model catalog from /v1/models",
		handler: async (_args, ctx) => {
			const result = await ctx.modelRegistry.refresh({
				providers: [REFUSELESS_PROVIDER_ID],
				allowNetwork: true,
				force: true,
			});

			const error = result.errors.get(REFUSELESS_PROVIDER_ID);
			if (error) {
				ctx.ui.notify(`Refuseless refresh failed: ${error.message}`, "error");
				return;
			}

			const count = ctx.modelRegistry.getAll().filter((model) => model.provider === REFUSELESS_PROVIDER_ID).length;
			ctx.ui.notify(`Refuseless catalog refreshed (${count} model${count === 1 ? "" : "s"}).`, "info");
		},
	});
}
