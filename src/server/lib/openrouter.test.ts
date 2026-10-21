import { describe, expect, it } from "vitest";
import { resolveChatAgentConfig } from "@/server/lib/openrouter";

const read =
  (env: Record<string, string>) =>
  (name: string): string | undefined =>
    env[name];

describe("resolveChatAgentConfig", () => {
  // Hosted billing reads OpenRouter's reported cost, so OpenRouter must win
  // whenever its key is present.
  it("prefers OpenRouter when both keys are set", () => {
    expect(
      resolveChatAgentConfig(
        read({ OPENROUTER_API_KEY: "or-key", OPENAI_API_KEY: "oa-key" }),
      ),
    ).toMatchObject({ provider: "openrouter", apiKey: "or-key" });
  });

  it("falls back to OpenAI with its model and base URL", () => {
    expect(
      resolveChatAgentConfig(
        read({
          OPENAI_API_KEY: "oa-key",
          OPENAI_MODEL: "gpt-x",
          OPENAI_BASE_URL: "http://localhost:11434/v1",
        }),
      ),
    ).toEqual({
      provider: "openai",
      apiKey: "oa-key",
      model: "gpt-x",
      baseURL: "http://localhost:11434/v1",
    });
  });

  it("throws when no key is set", () => {
    expect(() => resolveChatAgentConfig(read({}))).toThrow(
      "OPENROUTER_API_KEY or OPENAI_API_KEY",
    );
  });
});
