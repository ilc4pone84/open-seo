import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import {
  createOpenRouter,
  type LanguageModelV3,
} from "@openrouter/ai-sdk-provider";
import { getOptionalEnvValue } from "@/server/lib/runtime-env";

// OpenRouter model slug used for the SAM in-app chat agent. Override with
// OPENROUTER_MODEL to swap models without a code change.
const DEFAULT_CHAT_AGENT_MODEL = "openai/gpt-5.6-luna";

// Previous default; kept reachable via OPENROUTER_MODEL for rollback. Its
// routing needs the ZDR/provider tuning below.
const MINIMAX_M3 = "minimax/minimax-m3";

// Self-hosted alternative to OpenRouter: OPENAI_API_KEY talks to OpenAI (or,
// with OPENAI_BASE_URL, any OpenAI-compatible endpoint) directly. Usage cost
// is not reported on this path, so nothing is metered — hosted always runs on
// OpenRouter.
const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_OPENAI_MODEL = DEFAULT_CHAT_AGENT_MODEL.replace(/^openai\//, "");

const CHAT_AGENT_ENV_NAMES = [
  "OPENROUTER_API_KEY",
  "OPENROUTER_MODEL",
  "OPENAI_API_KEY",
  "OPENAI_MODEL",
  "OPENAI_BASE_URL",
] as const;

type ChatAgentEnvName = (typeof CHAT_AGENT_ENV_NAMES)[number];

type ChatAgentConfig = {
  provider: "openrouter" | "openai";
  apiKey: string;
  model?: string;
  baseURL?: string;
};

/**
 * Picks the LLM provider from env: OPENROUTER_API_KEY wins, OPENAI_API_KEY is
 * the fallback, and neither means AI features are not configured. Sync so the
 * SAM agent can resolve it from its DO env inside Think's sync `getModel()`.
 */
export function resolveChatAgentConfig(
  read: (name: ChatAgentEnvName) => string | undefined,
): ChatAgentConfig {
  const openrouterKey = read("OPENROUTER_API_KEY");
  if (openrouterKey) {
    return {
      provider: "openrouter",
      apiKey: openrouterKey,
      model: read("OPENROUTER_MODEL"),
    };
  }
  const openaiKey = read("OPENAI_API_KEY");
  if (openaiKey) {
    return {
      provider: "openai",
      apiKey: openaiKey,
      model: read("OPENAI_MODEL"),
      baseURL: read("OPENAI_BASE_URL"),
    };
  }
  throw new Error(
    "Missing required environment variable: OPENROUTER_API_KEY or OPENAI_API_KEY",
  );
}

export async function getChatAgentConfig(): Promise<ChatAgentConfig> {
  const values = new Map<ChatAgentEnvName, string | undefined>();
  for (const name of CHAT_AGENT_ENV_NAMES) {
    values.set(name, await getOptionalEnvValue(name));
  }
  return resolveChatAgentConfig((name) => values.get(name));
}

/**
 * Returns the AI SDK LanguageModel for the chat agent. On OpenRouter,
 * `usage: { include: true }` turns on usage accounting so each response
 * carries its real USD cost (providerMetadata.openrouter.usage.cost) — which
 * we meter against the shared usage-credit pool.
 *
 * Default model: GPT-5.6 Luna at `reasoning.effort: "max"` — "max" is valid at
 * the OpenRouter API for GPT-5.x but missing from the SDK's effort union, so
 * the reasoning config rides in `extraBody`. Reasoning tokens stream on the
 * separate reasoning channel and are billed as output tokens, which the usage
 * accounting above captures.
 *
 * Sync on purpose: Think's `getModel()` hook is sync and runs on every turn,
 * so the SAM agent reads the key/model from its DO env and builds here.
 */
export function buildChatAgentModel(
  config: ChatAgentConfig,
  reasoningEffort: "max" | "low" = "max",
): LanguageModelV3 {
  if (config.provider === "openai") {
    const openai = createOpenAICompatible({
      name: "openai",
      apiKey: config.apiKey,
      baseURL: config.baseURL ?? DEFAULT_OPENAI_BASE_URL,
      includeUsage: true,
      supportsStructuredOutputs: true,
      // OpenAI's reasoning models reject `max_tokens` and have no "max"
      // effort, so rename the cap and map the effort to its highest level.
      transformRequestBody: (args) => {
        const { max_tokens: maxTokens, ...body }: Record<string, unknown> =
          args;
        return {
          ...body,
          max_completion_tokens: maxTokens,
          reasoning_effort: reasoningEffort === "max" ? "high" : "low",
        };
      },
    });
    return openai(config.model ?? DEFAULT_OPENAI_MODEL);
  }

  const model = config.model ?? DEFAULT_CHAT_AGENT_MODEL;
  const openrouter = createOpenRouter({ apiKey: config.apiKey });

  // MiniMax M3 (env-override path only): `provider.order` prefers Together,
  // then Atlas Cloud (fp8); `zdr: true` restricts routing to Zero-Data-
  // Retention endpoints, which excludes MiniMax first-party — the account's
  // "Non-frontier requires ZDR" data policy enforces the same, this flag is
  // belt-and-braces. Fallbacks stay on within the ZDR set because pinning
  // providers caused a prod outage (Jul 2026: Together upstream-rate-limited
  // m3 and every chat turn 429'd). The explicit reasoning channel keeps m3's
  // `<think>` trace out of the visible answer text.
  if (model === MINIMAX_M3) {
    return openrouter(model, {
      usage: { include: true },
      reasoning: { effort: reasoningEffort === "low" ? "low" : "medium" },
      provider: {
        order: ["together", "atlas-cloud/fp8"],
        zdr: true,
        allow_fallbacks: true,
      },
    });
  }

  return openrouter(model, {
    usage: { include: true },
    extraBody: { reasoning: { effort: reasoningEffort } },
  });
}
