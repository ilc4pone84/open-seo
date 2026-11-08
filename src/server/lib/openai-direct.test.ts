import { expect, it } from "vitest";
import { z } from "zod";
import { isOpenAiDirect, toOpenAiBody } from "./openai-direct";

const translate = (body: object): unknown =>
  z.unknown().parse(JSON.parse(toOpenAiBody(JSON.stringify(body))));

it("only treats api.openai.com as direct OpenAI", () => {
  expect(isOpenAiDirect("https://api.openai.com/v1")).toBe(true);
  expect(isOpenAiDirect("https://llm.example.com/v1")).toBe(false);
  expect(isOpenAiDirect(undefined)).toBe(false);
  expect(isOpenAiDirect("not a url")).toBe(false);
});

it("translates OpenRouter fields to what OpenAI accepts", () => {
  expect(
    translate({
      model: "m",
      usage: { include: true },
      reasoning: { effort: "max" },
      max_tokens: 10,
      provider: { zdr: true },
      messages: [],
    }),
  ).toEqual({
    model: "m",
    reasoning_effort: "high",
    max_completion_tokens: 10,
    messages: [],
  });
});

it("keeps lower reasoning efforts and omits unset fields", () => {
  expect(translate({ reasoning: { effort: "low" } })).toEqual({
    reasoning_effort: "low",
  });
  expect(translate({ model: "m" })).toEqual({ model: "m" });
});

it("turns reasoning off with tools and relaxes strict schemas", () => {
  expect(
    translate({
      tools: [{}],
      reasoning: { effort: "max" },
      response_format: { json_schema: { strict: true, name: "r" } },
    }),
  ).toEqual({
    tools: [{}],
    reasoning_effort: "none",
    response_format: { json_schema: { strict: false, name: "r" } },
  });
});
