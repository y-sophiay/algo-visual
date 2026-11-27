import { callTool, ToolError } from "../tool";
import { findTool, tools } from "../tools";
import { chat, type Message, type ToolCall, type ToolMessage } from "./client";
import { SYSTEM_PROMPT } from "./prompt";
import { toOllamaTools } from "./to-ollama-tools";

/** The model's final answer, and the whole conversation so far so a follow-up question can continue it. */
export type AskResult = { answer: string; messages: Message[] };

/**
 * Answers one question with the local model. `history` is the conversation so far (empty for a first question):
 * pass the `messages` of an earlier answer to continue it. The system prompt is always the current SYSTEM_PROMPT,
 * sent once at the start; any system message already in `history` is replaced rather than repeated.
 *
 * The model can call the working tools. When it does, each call runs and its result goes back to the model, which
 * then answers from it. This is one round only, so the second request leaves the tools out: measured on qwen3:8b
 * (think off), the model answered from the result 3 times out of 3 without the tools and gave an empty reply 3 times
 * out of 3 with them (think on: 1 answer, 1 empty, 1 confused reply out of 3).
 *
 * TODO, one issue at a time. Each step has an `it.todo` in __tests__/ask.test.ts to turn into a real test.
 *
 * #11, ask one question (done): 1. build the messages, 2. one `chat` call, return the answer and the messages.
 *
 * #12, let the model call a tool (done): 3. pass the tools, 4. run each tool call and add its result, 5. call
 * `chat` again, without the tools, so the model answers from the results.
 *
 * #13, keep going until the model answers:
 *  6. Repeat steps 4 and 5 until a reply has no `tool_calls`, at most MAX_TURNS times (say 8). If it hits the
 *     limit, return an answer that says so instead of looping forever. The loop needs the tools in every request,
 *     so first find out why qwen3:8b replies empty when they come with a tool result (see above).
 */
export async function ask(question: string, history: Message[] = []): Promise<AskResult> {
  const messages: Message[] = [
    { role: "system", content: SYSTEM_PROMPT },
    ...history.filter((message) => message.role !== "system"),
    { role: "user", content: question },
  ];
  let reply = await chat({ messages, tools: toOllamaTools() });
  messages.push(reply);
  if (reply.tool_calls?.length) {
    messages.push(...reply.tool_calls.map(runToolCall));
    reply = await chat({ messages });
    messages.push(reply);
  }
  return { answer: reply.content, messages };
}

/**
 * Runs one tool call and returns its result as a tool message. Mistakes the model can fix (an unknown tool, bad
 * arguments, a ToolError) go back to it as the message's content instead of crashing; any other error is a bug and
 * is thrown.
 */
function runToolCall(call: ToolCall): ToolMessage {
  const name = call.function.name;
  const tool = findTool(name);
  if (!tool) {
    const names = tools.map((known) => known.name).join(", ");
    return { role: "tool", tool_name: name, content: `Unknown tool "${name}". The tools are: ${names}.` };
  }
  try {
    return { role: "tool", tool_name: name, content: JSON.stringify(callTool(tool, call.function.arguments)) };
  } catch (error) {
    if (!(error instanceof ToolError)) throw error;
    return { role: "tool", tool_name: name, content: error.message };
  }
}
