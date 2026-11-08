// Fork patch: lets OPENROUTER_BASE_URL point straight at api.openai.com.
// The OpenRouter provider sends fields OpenAI rejects, so translate them here
// and keep the rest of the AI code identical to upstream.
import { z } from "zod";

const requestSchema = z.looseObject({
  reasoning: z.object({ effort: z.string().optional() }).optional(),
  max_tokens: z.number().optional(),
  response_format: z
    .looseObject({ json_schema: z.looseObject({}).optional() })
    .optional(),
});

const OPENROUTER_ONLY = ["usage", "provider", "plugins", "models", "route"];

export function isOpenAiDirect(baseURL: string | undefined): boolean {
  if (!baseURL) return false;
  try {
    return new URL(baseURL).hostname === "api.openai.com";
  } catch {
    return false;
  }
}

export function toOpenAiBody(raw: string): string {
  const {
    reasoning,
    max_tokens: maxTokens,
    ...parsed
  } = requestSchema.parse(JSON.parse(raw));
  const body: Record<string, unknown> = Object.fromEntries(
    Object.entries(parsed).filter(([key]) => !OPENROUTER_ONLY.includes(key)),
  );
  const effort = reasoning?.effort;
  // OpenAI refuses function tools with reasoning on Chat Completions.
  if (body.tools) body.reasoning_effort = "none";
  else if (effort) body.reasoning_effort = effort === "max" ? "high" : effort;
  // OpenAI's strict mode requires every property; our schemas have optional ones.
  if (parsed.response_format?.json_schema) {
    parsed.response_format.json_schema.strict = false;
  }
  if (maxTokens !== undefined) body.max_completion_tokens = maxTokens;
  return JSON.stringify(body);
}

export const openAiDirectFetch: typeof fetch = (input, init) =>
  fetch(
    input,
    typeof init?.body === "string"
      ? { ...init, body: toOpenAiBody(init.body) }
      : init,
  );
