import { afterEach, describe, expect, it, vi } from "vitest";
import { callTool } from "../../tool";
import { runAlgorithmTool } from "../../tools";
import { ask } from "../ask";
import type { Message } from "../client";
import { OLLAMA } from "../config";
import { SYSTEM_PROMPT } from "../prompt";
import { toOllamaTools } from "../to-ollama-tools";
import { fakeOllama, toolCall } from "./fake-ollama";

// Each issue turns its it.todo lines into real tests. `fakeOllama` (./fake-ollama) gives the model's replies, so
// these never need Ollama running; client.test.ts shows how to use it.

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ask", () => {
  // #11
  it("sends the system prompt, the history and the question, and returns the model's answer", async () => {
    const history: Message[] = [
      { role: "user", content: "What is a stable sort?" },
      { role: "assistant", content: "One that keeps equal values in their original order." },
    ];
    const { requests } = fakeOllama(["Yes, merge sort is stable."]);

    const result = await ask("Is merge sort stable?", history);

    const sent: Message[] = [{ role: "system", content: SYSTEM_PROMPT }, ...history, { role: "user", content: "Is merge sort stable?" }];
    expect(requests).toEqual([
      { path: "/api/chat", body: { model: OLLAMA.model, messages: sent, tools: toOllamaTools(), think: OLLAMA.think, stream: false } },
    ]);
    expect(result.answer).toBe("Yes, merge sort is stable.");
    expect(result.messages).toEqual([...sent, { role: "assistant", content: "Yes, merge sort is stable." }]);
  });

  it("sends the system prompt once when the history comes from an earlier answer", async () => {
    const { requests } = fakeOllama(["One that keeps equal values in their original order.", "Yes."]);

    const first = await ask("What is a stable sort?");
    await ask("Is merge sort stable?", first.messages);

    const sent = requests[1].body?.messages ?? [];
    expect(sent.filter((message) => message.role === "system")).toEqual([{ role: "system", content: SYSTEM_PROMPT }]);
    expect(sent.map((message) => message.role)).toEqual(["system", "user", "assistant", "user"]);
  });

  // #12
  it("runs the tool the model asks for and sends the result back as a tool message", async () => {
    const args = { slug: "quick-sort", input: { kind: "array", size: 50, distribution: "reversed" } };
    const { requests } = fakeOllama([toolCall("run_algorithm", args), "Quick Sort made 133 swaps."]);

    const result = await ask("How many swaps does quick sort make on 50 reversed numbers?");

    // The engine's result, the same as running the tool directly.
    const output = callTool(runAlgorithmTool, args);
    const sent: Message[] = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: "How many swaps does quick sort make on 50 reversed numbers?" },
      { role: "assistant", content: "", ...toolCall("run_algorithm", args) },
      { role: "tool", tool_name: "run_algorithm", content: JSON.stringify(output) },
    ];
    expect(requests).toHaveLength(2);
    // The tools go with the question; the second request asks for an answer, so it leaves them out.
    expect(requests.map((request) => request.body?.tools)).toEqual([toOllamaTools(), undefined]);
    expect(requests[1].body?.messages).toEqual(sent);
    expect(result.answer).toBe("Quick Sort made 133 swaps.");
    expect(result.messages).toEqual([...sent, { role: "assistant", content: "Quick Sort made 133 swaps." }]);
  });

  it("sends a ToolError or an unknown tool name back to the model instead of throwing", async () => {
    const { requests } = fakeOllama([
      toolCall("run_algorithm", { slug: "slow-sort" }),
      "There is no slow-sort.",
      toolCall("sort_numbers", { values: [3, 1, 2] }),
      "I can't sort numbers directly.",
    ]);

    const badSlug = await ask("How many swaps does slow sort make?");
    const unknownTool = await ask("Sort 3, 1, 2 for me.");

    // The model sees what went wrong, so it can fix its call or explain.
    expect(requests[1].body?.messages.at(-1)).toEqual({
      role: "tool",
      tool_name: "run_algorithm",
      content: 'Unknown algorithm "slow-sort". list_algorithms gives every slug.',
    });
    expect(requests[3].body?.messages.at(-1)).toEqual({
      role: "tool",
      tool_name: "sort_numbers",
      content: 'Unknown tool "sort_numbers". The tools are: run_algorithm, make_link.',
    });
    expect(badSlug.answer).toBe("There is no slow-sort.");
    expect(unknownTool.answer).toBe("I can't sort numbers directly.");
  });

  // #13
  it.todo("keeps running tools until the model answers in words");
  it.todo("runs every tool call in one reply");
  it.todo("stops after MAX_TURNS and says so");
});

describe("terminal chat", () => {
  // #14
  it.todo("passes the earlier messages as history, so a follow-up question continues the conversation");
});
