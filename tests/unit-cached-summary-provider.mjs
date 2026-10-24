import assert from "node:assert/strict";
import { mock, test } from "node:test";
import childProcess from "node:child_process";
import { syncBuiltinESMExports } from "node:module";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import activate, { __test } from "../src/index.js";
import { CachedSummaries } from "../src/cached-summary.js";
import { QueryContext } from "../src/query-state.js";

const model = { id: "synthetic", provider: "claude-bridge", api: "claude-bridge", baseUrl: "claude-bridge" };
const handlers = new Map();
let provider;
activate({ on: (name, fn) => handlers.set(name, fn), registerProvider: (_name, config) => { provider = config; }, registerTool: () => {} });

test("the exact fork header wins over standalone retention-none routing", async () => {
	let called = 0, spawns = 0;
	const output = { role: "assistant", api: model.api, provider: model.provider, model: model.id, content: [{ type: "text", text: "FORK" }], stopReason: "stop", timestamp: 1,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
	const summary = mock.method(CachedSummaries.prototype, "stream", (m, context, options) => {
		called++; assert.equal(m, model); assert.equal(context.messages[0].content, "Summarize"); assert.equal(options.cacheRetention, "none");
		const stream = createAssistantMessageEventStream(); stream.push({ type: "done", reason: "stop", message: output }); stream.end(); return stream;
	});
	const noSpawn = mock.method(childProcess, "spawn", () => { spawns++; throw new Error("Offline provider test forbids native execution"); });
	syncBuiltinESMExports();
	try {
		assert(provider);
		const result = await provider.streamSimple(model, { messages: [{ role: "user", content: "Summarize", timestamp: 1 }] }, { cacheRetention: "none", headers: { "x-claude-bridge-fork": "1" }, sessionId: "fixture" }).result();
		assert.equal(result, output); assert.equal(called, 1); assert.equal(spawns, 0);
	} finally { summary.mock.restore(); noSpawn.mock.restore(); syncBuiltinESMExports(); }
});

test("native init captures the first-query source before a shared mirror or open Pi stream exists", async () => {
	const context = new QueryContext();
	assert.equal(context.currentPiStream, null);
	const capture = { valid: true };
	const query = (async function* () { yield { type: "system", subtype: "init", session_id: "first-native-id" }; })();
	await __test.consumeQuery(query, new Map(), model, () => false, context, capture);
	assert.equal(capture.sessionId, "first-native-id");
});

test("bridge compaction observer does not claim ownership or produce its own summary", () => {
	const result = handlers.get("session_before_compact")({ signal: new AbortController().signal }, { sessionManager: { getSessionId: () => "fixture" } });
	assert.equal(result, undefined);
});
