import assert from "node:assert/strict";
import { after, test } from "node:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createSession, getSessionPath, openSession } from "cc-session-io";
import { CachedSummaries } from "../src/cached-summary.js";
import { createToolServer } from "../src/mcp-server.js";
import { rememberSessionCwd, forgetSessionCwd } from "../src/session-cwd.js";

const root = mkdtempSync(join(tmpdir(), "bridge-summary-stream-"));
after(() => rmSync(root, { recursive: true, force: true }));
const model = {
	id: "synthetic-claude", provider: "claude-bridge", api: "claude-bridge", baseUrl: "claude-bridge",
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, maxTokens: 8192, contextWindow: 200000,
};
const usage = { input_tokens: 100, output_tokens: 41, cache_read_input_tokens: 3000, cache_creation_input_tokens: 0 };
const tool = { name: "read", description: "Synthetic read", parameters: {
	type: "object", anyOf: [
		{ properties: { path: { type: "string" } }, required: ["path"] },
		{ properties: { id: { type: "integer" } }, required: ["id"] },
	],
} };
function output(m, text, reason, error) {
	return { role: "assistant", api: m.api, provider: m.provider, model: m.id, content: text ? [{ type: "text", text }] : [], stopReason: reason, errorMessage: error, timestamp: 1,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
}
function fakeSdk() {
	const calls = [], control = { policy: { effective: {}, sources: [], provenance: {} }, policyCalls: 0 };
	return { calls, control, sdk: {
		async resolveSettings() { control.policyCalls++; await control.duringPolicy?.(); return control.policy; },
		query(input) {
			const options = input.options;
			const path = getSessionPath(options.resume, options.cwd);
			const entry = { ...input, path, records: structuredClone(openSession({ sessionId: options.resume, projectPath: options.cwd }).records) };
			calls.push(entry);
			const child = options.spawnClaudeCodeProcess({ command: process.execPath, args: ["-e", "setTimeout(()=>process.exit(0),120)"], cwd: options.cwd, env: options.env, signal: new AbortController().signal });
			entry.child = child;
			const iterator = (async function* () {
				if (control.throwQuery) throw new Error("Synthetic summary failure");
				yield { type: "assistant", message: { content: control.toolCall ? [{ type: "tool_use", id: "forbidden", name: "mcp__custom-tools__read", input: { path: "DO_NOT_READ" } }] : [{ type: "text", text: "CHECKPOINT" }], usage, stop_reason: control.assistantStop ?? "end_turn" } };
				yield { type: "result", subtype: control.errorResult ? "error_during_execution" : "success", is_error: !!control.errorResult, result: "CHECKPOINT", stop_reason: control.stopReason ?? "end_turn", session_id: control.reportedId, usage };
			})();
			iterator.close = () => child.stdin?.end();
			iterator.interrupt = async () => {};
			return iterator;
		},
	} };
}
function fixture(cache, id = "test") {
	const cwd = mkdtempSync(join(root, id + "-"));
	rememberSessionCwd(id, cwd);
	const native = createSession({ projectPath: cwd });
	const first = "EARLIER_CEDAR_" + id;
	native.addUserMessage(first); native.addAssistantMessage([{ type: "text", text: "READY" }]); native.save();
	const context = { systemPrompt: "SYNTHETIC_SYSTEM", messages: [{ role: "user", content: first, timestamp: 1 }], tools: [structuredClone(tool)] };
	const queryOptions = { cwd, resume: native.sessionId, extraArgs: { "strict-mcp-config": null, model: model.id }, tools: [], env: { ...process.env }, settings: {}, systemPrompt: { type: "preset", preset: "claude_code", append: "ORIGINAL_APPEND" }, effort: "low" };
	const capture = cache.record(id, model, context, queryOptions, [tool], "low");
	const sent = { ...context, messages: [...context.messages, { role: "assistant", content: [{ type: "text", text: "READY" }], provider: "claude-bridge", timestamp: 2 }, { role: "user", content: [{ type: "text", text: "Summarize." }], timestamp: 3 }] };
	return { id, cwd, native, context, capture, queryOptions, sent, options: { sessionId: id, reasoning: "low", cacheRetention: "none", maxTokens: 1024 } };
}
async function cleanup(cache, ...fixtures) {
	await cache.shutdown();
	for (const f of fixtures) forgetSessionCwd(f.id);
}

test("private query keeps native options and normalized tool definitions, refuses execution, and drains before delivery", async () => {
	const fake = fakeSdk(), cache = new CachedSummaries(output, fake.sdk), f = fixture(cache);
	const before = readFileSync(f.native.jsonlPath), mtime = statSync(f.native.jsonlPath).mtimeMs;
	try {
		const response = await cache.stream(model, f.sent, f.options).result();
		assert.equal(response.stopReason, "stop"); assert.equal(response.usage.output, 41); assert.equal(response.usage.cacheRead, 3000);
		assert.equal(fake.calls.length, 1);
		const call = fake.calls[0], q = call.options;
		assert.notEqual(q.resume, f.native.sessionId); assert.equal(call.prompt, "Summarize.");
		assert.equal(q.cwd, f.cwd); assert.notEqual(q.cwd, process.cwd());
		assert.deepEqual(q.systemPrompt, f.queryOptions.systemPrompt); assert.equal(q.effort, "low");
		assert.equal(q.settings.disableAllHooks, true); assert.equal(q.settings.autoMemoryEnabled, false);
		assert.equal(q.strictMcpConfig, true); assert.equal(q.persistSession, false); assert.equal(q.maxTurns, 1);
		assert.equal(q.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS, "1024");
		const server = q.mcpServers["custom-tools"].instance.server;
		const listed = await server._requestHandlers.get("tools/list")({ method: "tools/list" }, {});
		const normal = createToolServer("custom-tools", [{ name: tool.name, description: tool.description, inputSchema: tool.parameters, handler: () => { throw new Error("A real handler must not run"); } }]);
		const expected = await normal.instance.server._requestHandlers.get("tools/list")({ method: "tools/list" }, {});
		assert.deepEqual(listed, expected);
		const denied = await server._requestHandlers.get("tools/call")({ method: "tools/call", params: { name: "read", arguments: { path: "DO_NOT_READ" }, _meta: { "claudecode/toolUseId": "denied" } } }, {});
		assert.equal(denied.isError, true); assert.match(denied.content[0].text, /cannot execute/);
		assert.equal(call.child.exitCode, 0); assert(!existsSync(call.path));
		assert.deepEqual(readFileSync(f.native.jsonlPath), before); assert.equal(statSync(f.native.jsonlPath).mtimeMs, mtime);
	} finally { await cleanup(cache, f); }
});

test("metadata and unsafe-policy failures decline before a query", async () => {
	for (const kind of ["missing", "model", "effort", "history", "system", "tools", "cwd", "stale", "strict-mcp", "managed-hooks", "managed-mcp", "policy-helper", "cloud-mcp", "pre-aborted"]) {
		const fake = fakeSdk(), cache = new CachedSummaries(output, fake.sdk), f = fixture(cache);
		let chosen = model, context = structuredClone(f.sent), options = { ...f.options };
		try {
			if (kind === "missing") options.sessionId = "missing";
			if (kind === "model") chosen = { ...model, id: "different" };
			if (kind === "effort") options.reasoning = "high";
			if (kind === "history") context.messages[0].content = "different";
			if (kind === "system") context.systemPrompt = "different";
			if (kind === "tools") context.tools[0].description = "different";
			if (kind === "cwd") rememberSessionCwd(f.id, root);
			if (kind === "stale") cache.invalidate(f.id);
			if (kind === "strict-mcp") delete f.capture.options.extraArgs["strict-mcp-config"];
			if (kind === "managed-hooks") fake.control.policy.sources = [{ source: "managed", settings: { hooks: { SessionStart: [] } } }];
			if (kind === "managed-mcp") fake.control.policy.sources = [{ source: "managed", settings: { managedMcpServers: { unsafe: {} } } }];
			if (kind === "policy-helper") fake.control.policy.sources = [{ source: "user", settings: { policyHelper: "DO_NOT_RUN" } }];
			if (kind === "cloud-mcp") fake.control.policy.effective = { env: { ENABLE_CLAUDEAI_MCP_SERVERS: "1" } };
			if (kind === "pre-aborted") { const c = new AbortController(); c.abort(); options.signal = c.signal; }
			const response = await cache.stream(chosen, context, options).result();
			assert.notEqual(response.stopReason, "stop", kind); assert.equal(fake.calls.length, 0, kind);
		} finally { await cleanup(cache, f); }
	}
});

test("configuration directory drift before or during policy resolution refuses without a query", async () => {
	for (const timing of ["before", "during"]) {
		const fake = fakeSdk(), cache = new CachedSummaries(output, fake.sdk), f = fixture(cache);
		const original = process.env.CLAUDE_CONFIG_DIR, changed = join(root, "changed-" + timing); mkdirSync(changed, { recursive: true });
		try {
			if (timing === "before") process.env.CLAUDE_CONFIG_DIR = changed;
			else fake.control.duringPolicy = () => { process.env.CLAUDE_CONFIG_DIR = changed; };
			const response = await cache.stream(model, f.sent, f.options).result();
			assert.equal(response.stopReason, "error"); assert.equal(fake.calls.length, 0);
			if (timing === "before") assert.equal(fake.control.policyCalls, 0);
		} finally { process.env.CLAUDE_CONFIG_DIR = original; await cleanup(cache, f); }
	}
});

for (const origin of ["result", "assistant"]) test(`token limit in ${origin} returns length rather than a completed checkpoint`, async () => {
	const fake = fakeSdk(), cache = new CachedSummaries(output, fake.sdk), f = fixture(cache);
	try {
		fake.control[origin === "result" ? "stopReason" : "assistantStop"] = "max_tokens";
		const response = await cache.stream(model, f.sent, f.options).result();
		assert.equal(response.stopReason, "length"); assert.equal(response.usage.output, 41); assert(!existsSync(fake.calls[0].path));
	} finally { await cleanup(cache, f); }
});

for (const mode of ["failure", "tool-call", "throw"]) test(`summary ${mode} preserves observed usage and only cleans its owned file`, async () => {
	const fake = fakeSdk(), cache = new CachedSummaries(output, fake.sdk), f = fixture(cache);
	const before = readFileSync(f.native.jsonlPath);
	try {
		fake.control.reportedId = f.native.sessionId;
		fake.control.errorResult = mode === "failure"; fake.control.toolCall = mode === "tool-call"; fake.control.throwQuery = mode === "throw";
		const response = await cache.stream(model, f.sent, f.options).result();
		assert.equal(response.stopReason, "error"); assert.equal(response.usage.output, mode === "throw" ? 0 : 41);
		assert.deepEqual(readFileSync(f.native.jsonlPath), before); assert(!existsSync(fake.calls[0].path));
	} finally { await cleanup(cache, f); }
});

test("capture replacement during child cleanup invalidates the answer", async () => {
	const fake = fakeSdk(), cache = new CachedSummaries(output, fake.sdk), f = fixture(cache);
	let timer;
	try {
		const pending = cache.stream(model, f.sent, f.options).result();
		timer = setTimeout(() => cache.record(f.id, model, f.context, f.queryOptions, [tool], "low"), 35);
		const response = await pending; assert.equal(response.stopReason, "error"); assert.match(response.errorMessage, /stale/); assert.equal(response.usage.output, 41);
	} finally { clearTimeout(timer); await cleanup(cache, f); }
});

test("abort remains effective while the child exits", async () => {
	const fake = fakeSdk(), cache = new CachedSummaries(output, fake.sdk), f = fixture(cache), controller = new AbortController();
	let timer;
	try {
		const pending = cache.stream(model, f.sent, { ...f.options, signal: controller.signal }).result();
		timer = setTimeout(() => controller.abort(), 35);
		const response = await pending; assert.equal(response.stopReason, "aborted"); assert.equal(response.usage.output, 41); assert(!existsSync(fake.calls[0].path));
	} finally { clearTimeout(timer); await cleanup(cache, f); }
});

test("different sessions retain separate cwd and capture ownership during concurrent summaries", async () => {
	const fake = fakeSdk(), cache = new CachedSummaries(output, fake.sdk), a = fixture(cache, "A"), b = fixture(cache, "B");
	try {
		const [ra, rb] = await Promise.all([cache.stream(model, a.sent, a.options).result(), cache.stream(model, b.sent, b.options).result()]);
		assert.equal(ra.stopReason, "stop"); assert.equal(rb.stopReason, "stop"); assert.equal(fake.calls.length, 2);
		assert.deepEqual(fake.calls.map(c => c.options.cwd).sort(), [a.cwd, b.cwd].sort());
		assert.notEqual(fake.calls[0].options.resume, fake.calls[1].options.resume);
		for (const call of fake.calls) assert(!existsSync(call.path));
	} finally { await cleanup(cache, a, b); }
});

test("shutdown cancels and drains outstanding private work", async () => {
	const fake = fakeSdk(), cache = new CachedSummaries(output, fake.sdk), f = fixture(cache);
	let timer;
	try {
		const pending = cache.stream(model, f.sent, f.options).result();
		timer = setTimeout(() => void cache.shutdown(), 35);
		const response = await pending; assert.equal(response.stopReason, "aborted"); await cache.shutdown(); assert(!existsSync(fake.calls[0].path));
	} finally { clearTimeout(timer); await cleanup(cache, f); }
});
