// Native transcript forks for pi-cached-compaction. Ordinary summaries stay on Pi's path.
import { createHash, randomUUID } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { query, resolveSettings } from "@anthropic-ai/claude-agent-sdk";
import { createAssistantMessageEventStream, type AssistantMessage, type Context, type Model, type SimpleStreamOptions, type Tool } from "@earendil-works/pi-ai";
import { deleteSession, getClaudeDir, getSessionPath, openSession, type UserContentBlock, type ContentBlock } from "cc-session-io";
import { convertPiMessages } from "./convert.js";
import { createToolServer } from "./mcp-server.js";
import { MCP_SERVER_NAME, MCP_TOOL_PREFIX } from "./skills.js";
import { updateUsage } from "./usage.js";
import { sessionCwdFor } from "./session-cwd.js";

type QueryOptions = NonNullable<Parameters<typeof query>[0]["options"]>;
type Output = (model: Model<any>, text: string, reason: AssistantMessage["stopReason"], error?: string) => AssistantMessage;
type SummarySdk = { query: typeof query; resolveSettings: typeof resolveSettings };
export interface SummaryCapture {
	context: Context;
	model: Model<any>;
	options: QueryOptions;
	tools: Tool[];
	cwd: string;
	claudeDir?: string;
	environmentHash: string;
	reasoning?: SimpleStreamOptions["reasoning"];
	sessionId?: string;
	valid: boolean;
}
type RecordEntry = Record<string, any>;
type Atom = { role: string; block: RecordEntry };

function stable(value: any): string {
	return JSON.stringify(value, (_key, item) => item && typeof item === "object" && !Array.isArray(item)
		? Object.fromEntries(Object.keys(item).sort().map((key) => [key, item[key]])) : item);
}

function environmentHash(): string {
	return createHash("sha256").update(stable(process.env)).digest("hex");
}

function environmentMatches(capture: SummaryCapture): boolean {
	return capture.claudeDir === getClaudeDir() && capture.environmentHash === environmentHash();
}

function blocks(content: any): RecordEntry[] {
	if (typeof content === "string") return [{ type: "text", text: content }];
	if (!Array.isArray(content)) throw new Error("Unsupported native message content");
	return content.flatMap((b): RecordEntry[] => {
		switch (b.type) {
			case "thinking": return [];
			case "text": return b.text ? [{ type: "text", text: b.text }] : [];
			case "image": return [{ type: "image", source: b.source }];
			case "tool_use": return [{ type: "tool_use", id: b.id, name: b.name, input: b.input }];
			case "tool_result": return [{ type: "tool_result", tool_use_id: b.tool_use_id, content: blocks(b.content), is_error: b.is_error === true }];
			default: throw new Error("Unsupported native content block");
		}
	});
}

function atoms(messages: Array<{ role: string; content: any }>): Atom[] {
	return messages.flatMap((m) => blocks(m.content).map((block) => ({ role: m.role, block })));
}

function converted(messages: Context["messages"], names: Map<string, string>) {
	for (const message of messages) {
		if ((message.role === "user" || message.role === "toolResult") && Array.isArray(message.content)
			&& message.content.some((b) => b.type !== "text" && b.type !== "image")) throw new Error("Unsupported Pi content in native fork");
	}
	const result = convertPiMessages(messages, names);
	if (result.dropped.thinking || result.dropped.abortedTurns || result.dropped.other.size) {
		throw new Error("Native fork would omit Pi content");
	}
	return result.anthropicMessages;
}

/** Copy only a verified native prefix, keeping its UUIDs, thinking and attachments.
 * Pi-only suffixes are appended to the private copy, never to the live transcript. */
export function prepareSummaryFork(capture: SummaryCapture, history: Context["messages"]): { sessionId: string; prefixAtoms: number; suffixMessages: number } {
	const names = new Map(capture.tools.map((t) => [t.name, `${MCP_TOOL_PREFIX}${t.name}`]));
	const expected = atoms(converted(history, names));
	const linear = history.flatMap((message) => atoms(converted([message], names)));
	if (stable(expected) !== stable(linear)) throw new Error("Pi history needs tool-result reordering; decline a guessed fork cut");
	const base = atoms(converted(capture.context.messages, names));
	if (!base.length) throw new Error("No native request prefix to fork");
	const pending = new Set<string>();
	const used = new Set<string>();
	for (const { block } of expected) {
		if (block.type === "tool_use") {
			if (used.has(block.id)) throw new Error("Duplicate native tool call ID");
			used.add(block.id);
			pending.add(block.id);
		}
		if (block.type === "tool_result" && !pending.delete(block.tool_use_id)) throw new Error("Unpaired tool result in summary history");
	}
	if (pending.size) throw new Error("Summary history has unfinished tool calls");
	const boundaries = new Map<number, number>();
	let count = 0;
	history.forEach((message, i) => {
		count += atoms(converted([message], names)).length;
		boundaries.set(count, i + 1);
	});
	const source = getSessionPath(capture.sessionId!, capture.cwd, capture.claudeDir);
	const records: RecordEntry[] = readFileSync(source, "utf8").split("\n").filter((s) => s.trim()).map((s) => JSON.parse(s));
	const chained = records.filter((r) => r.uuid && r.isSidechain !== true && ["user", "assistant", "attachment", "system"].includes(r.type));
	const byId = new Map(chained.map((r) => [r.uuid, r]));
	const chain: RecordEntry[] = [];
	const seen = new Set<string>();
	let node = chained.at(-1);
	while (node) {
		if (seen.has(node.uuid)) throw new Error("Cyclic native transcript");
		seen.add(node.uuid);
		chain.unshift(node);
		if (!node.parentUuid) break;
		const parent = byId.get(node.parentUuid);
		if (!parent) throw new Error("Incomplete native transcript chain");
		node = parent;
	}
	let matched = 0;
	let cut: RecordEntry | undefined;
	let piCut = 0;
	let prefixAtoms = 0;
	for (const record of chain) {
		if (record.type === "attachment" || (record.type === "system" && record.subtype === "turn_duration")) {
			// Queued steers need attachment-to-Pi attribution, not a guessed match.
			if (record.attachment?.type === "queued_command") throw new Error("Queued native steer cannot be attributed safely");
			if (cut && matched === prefixAtoms) cut = record;
			continue;
		}
		if (record.type !== "user" && record.type !== "assistant") throw new Error("Unsupported native chain entry");
		const next = atoms([record.message]);
		if (next.some((item, i) => stable(item) !== stable(expected[matched + i]))) break;
		matched += next.length;
		if (matched >= base.length && boundaries.has(matched)) {
			cut = record;
			piCut = boundaries.get(matched)!;
			prefixAtoms = matched;
		}
	}
	if (!cut || prefixAtoms < base.length) throw new Error("Native transcript does not cover the captured request");
	const keep = new Set(chain.slice(0, chain.indexOf(cut) + 1).map((r) => r.uuid));
	const cutIndex = records.indexOf(cut);
	const sessionId = randomUUID();
	const target = getSessionPath(sessionId, capture.cwd, capture.claudeDir);
	const copied = records.slice(0, cutIndex + 1).filter((r) => !r.uuid || keep.has(r.uuid));
	writeFileSync(target, copied.map((r) => JSON.stringify({ ...r, sessionId })).join("\n") + "\n", { flag: "wx", mode: 0o600 });
	try {
		const suffix = converted(history.slice(piCut), names);
		if (suffix.length) {
			const fork = openSession({ sessionId, projectPath: capture.cwd, claudeDir: capture.claudeDir });
			// importMessages repairs results in isolation and would discard results whose
			// tool_use is already in the copied prefix. Append the validated blocks directly.
			for (const message of suffix) {
				if (message.role === "user") fork.addUserMessage(message.content as string | UserContentBlock[]);
				else fork.addAssistantMessage(typeof message.content === "string" ? [{ type: "text", text: message.content }] : message.content as ContentBlock[]);
			}
			fork.save();
		}
		const assembled = openSession({ sessionId, projectPath: capture.cwd, claudeDir: capture.claudeDir });
		if (stable(atoms(assembled.messages.map((r) => r.message))) !== stable(expected)) throw new Error("Assembled native fork differs from Pi history");
		return { sessionId, prefixAtoms, suffixMessages: suffix.length };
	} catch (error) {
		deleteSession(sessionId, capture.cwd, capture.claudeDir);
		throw error;
	}
}

/** SDK close() is not process exit. Do not delete a transcript a child can still write. */
export function watchedSummaryProcess() {
	let child: ChildProcess | undefined;
	let closed = false;
	let finish!: () => void;
	const exited = new Promise<void>((resolve) => { finish = resolve; });
	const start: NonNullable<QueryOptions["spawnClaudeCodeProcess"]> = (options) => {
		if (closed || options.signal.aborted) throw new Error("Summary process was cancelled before spawn");
		child = spawn(options.command, options.args, { cwd: options.cwd, env: options.env, stdio: ["pipe", "pipe", "ignore"], windowsHide: true });
		const proc = child;
		proc.once("close", finish);
		proc.once("error", () => { if (proc.pid === undefined) finish(); });
		options.signal.addEventListener("abort", () => { if (proc.exitCode === null && proc.signalCode === null) proc.kill(); }, { once: true });
		return proc as ReturnType<NonNullable<QueryOptions["spawnClaudeCodeProcess"]>>;
	};
	return {
		start,
		async close(): Promise<boolean> {
			closed = true;
			if (!child) finish();
			const kill = setTimeout(() => { try { child?.kill("SIGKILL"); } catch {} }, 2_000);
			let deadline: ReturnType<typeof setTimeout>;
			const stopped = await Promise.race([exited.then(() => true), new Promise<false>((resolve) => { deadline = setTimeout(() => resolve(false), 4_000); })]);
			clearTimeout(kill);
			clearTimeout(deadline!);
			return stopped;
		},
		exited,
	};
}

export class CachedSummaries {
	private captures = new Map<string, SummaryCapture>();
	private controllers = new Set<AbortController>();
	private jobs = new Set<Promise<void>>();
	constructor(private output: Output, private sdk: SummarySdk = { query, resolveSettings }) {}

	record(session: string | undefined, model: Model<any>, context: Context, options: QueryOptions, tools: Tool[], reasoning?: SimpleStreamOptions["reasoning"]): SummaryCapture | undefined {
		if (!session) return undefined;
		this.invalidate(session);
		try {
			const { mcpServers: _servers, stderr: _stderr, ...plain } = options;
			const capture: SummaryCapture = { ...structuredClone({ context, model, options: plain, tools }), cwd: options.cwd!, claudeDir: options.env?.CLAUDE_CONFIG_DIR ?? getClaudeDir(), environmentHash: environmentHash(), reasoning, sessionId: options.resume, valid: true };
			this.captures.set(session, capture);
			return capture;
		} catch {
			this.captures.delete(session);
			return undefined;
		}
	}

	invalidate(session?: string | null) {
		if (session) {
			const capture = this.captures.get(session);
			if (capture) capture.valid = false;
		} else for (const capture of this.captures.values()) capture.valid = false;
	}

	clear() {
		this.invalidate();
		this.captures.clear();
		for (const controller of this.controllers) controller.abort();
	}

	async shutdown() {
		this.clear();
		await Promise.allSettled([...this.jobs]);
	}

	stream(model: Model<any>, context: Context, options?: SimpleStreamOptions) {
		const stream = createAssistantMessageEventStream();
		const job = this.run(model, context, options, stream);
		this.jobs.add(job);
		void job.finally(() => this.jobs.delete(job));
		return stream;
	}

	private async run(model: Model<any>, context: Context, options: SimpleStreamOptions | undefined, stream: ReturnType<typeof createAssistantMessageEventStream>) {
		const controller = new AbortController();
		this.controllers.add(controller);
		const onAbort = () => controller.abort();
		options?.signal?.addEventListener("abort", onAbort, { once: true });
		if (options?.signal?.aborted) controller.abort();
		let fork: { sessionId: string } | undefined;
		let source: SummaryCapture | undefined;
		let sdkQuery: ReturnType<typeof query> | undefined;
		const response = this.output(model, "", "stop");
		const watched = watchedSummaryProcess();
		try {
			const capture = options?.sessionId ? this.captures.get(options.sessionId) : undefined;
			source = capture;
			const history = context.messages.slice(0, -1);
			const instruction = context.messages.at(-1);
			if (!capture?.valid || !capture.sessionId || controller.signal.aborted) throw new Error("No current native conversation for cached summary");
			if (!environmentMatches(capture) || capture.cwd !== sessionCwdFor(options?.sessionId) || stable(capture.model) !== stable(model) || capture.reasoning !== options?.reasoning
				|| capture.context.systemPrompt !== context.systemPrompt || stable(capture.context.tools) !== stable(context.tools)
				|| capture.context.messages.some((m, i) => stable(m) !== stable(history[i]))) throw new Error("Cached summary does not match its native request");
			const prompt = instruction?.role === "user" ? typeof instruction.content === "string" ? instruction.content
				: instruction.content.every((b) => b.type === "text") ? instruction.content.map((b) => (b as { text: string }).text).join("\n") : "" : "";
			if (!prompt.trim()) throw new Error("Invalid cached summary instruction");
			if (capture.options.extraArgs?.["strict-mcp-config"] === undefined) throw new Error("Cached summaries require strict MCP configuration");
			const policy = await this.sdk.resolveSettings({ cwd: capture.cwd, settingSources: capture.options.settingSources });
			const cloudMcp = policy.effective?.env?.ENABLE_CLAUDEAI_MCP_SERVERS;
			if (cloudMcp && !["0", "false", "no", "off"].includes(cloudMcp.toLowerCase())) throw new Error("Settings would re-enable cloud MCP in a summary");
			if (policy.sources.some((s) => (s.settings as RecordEntry).policyHelper || s.settings.managedMcpServers || (s.source === "managed" && (s.settings.hooks || (s.settings as RecordEntry).mcpServers)))) {
				throw new Error("Managed hooks, MCP or policy helpers prevent an isolated summary");
			}
			if (!environmentMatches(capture) || !capture.valid || this.captures.get(options!.sessionId!) !== capture || controller.signal.aborted) throw new Error("Cached summary source became stale");
			fork = prepareSummaryFork(capture, history);
			const mcpServers = capture.tools.length ? { [MCP_SERVER_NAME]: createToolServer(MCP_SERVER_NAME, capture.tools.map((tool) => ({
				name: tool.name, description: tool.description, inputSchema: tool.parameters,
				handler: async (toolCallId: string) => ({ toolCallId, isError: true, content: [{ type: "text" as const, text: "Tools cannot execute in a context summary." }] }),
			}))) } : undefined;
			sdkQuery = this.sdk.query({ prompt, options: {
				...capture.options,
				resume: fork.sessionId,
				mcpServers,
				settings: { ...(capture.options.settings as object), disableAllHooks: true, autoMemoryEnabled: false },
				strictMcpConfig: true,
				abortController: controller,
				maxTurns: 1,
				persistSession: false,
				spawnClaudeCodeProcess: watched.start,
				...(options?.maxTokens ? { env: { ...capture.options.env, CLAUDE_CODE_MAX_OUTPUT_TOKENS: String(options.maxTokens) } } : {}),
			} });
			let text = "";
			let final = "";
			let succeeded = false;
			let truncated = false;
			for await (const message of sdkQuery) {
				if (controller.signal.aborted) throw new Error("Summary aborted");
				if (message.type === "assistant") {
					updateUsage(response, message.message.usage, model);
					for (const block of message.message.content) {
						if (block.type === "tool_use") throw new Error("Summary attempted a tool call");
						if (block.type === "text") text += block.text;
					}
					if (message.message.stop_reason === "max_tokens") truncated = true;
				} else if (message.type === "stream_event") {
					const event = message.event;
					if (event.type === "message_start") updateUsage(response, event.message.usage, model);
					if (event.type === "message_delta") {
						updateUsage(response, event.usage, model);
						if (event.delta.stop_reason === "max_tokens") truncated = true;
					}
				} else if (message.type === "result") {
					updateUsage(response, message.usage, model);
					if (message.subtype !== "success" || message.is_error) throw new Error("Claude cached summary failed");
					succeeded = true;
					final = message.result;
					if (message.stop_reason === "max_tokens") truncated = true;
				}
			}
			if (!succeeded || !(final || text).trim()) throw new Error("Claude cached summary returned no completed text");
			if (!capture.valid || this.captures.get(options!.sessionId!) !== capture || controller.signal.aborted) throw new Error("Cached summary source became stale");
			response.content = [{ type: "text", text: final || text }];
			response.stopReason = truncated ? "length" : "stop";
		} catch (error) {
			response.stopReason = controller.signal.aborted ? "aborted" : "error";
			response.content = [];
			response.errorMessage = error instanceof Error ? error.message : String(error);
		} finally {
			try { sdkQuery?.close(); } catch {}
			const stopped = await watched.close();
			const cleanup = () => { if (fork && source) deleteSession(fork.sessionId, source.cwd, source.claudeDir); };
			try {
				if (stopped) cleanup();
				else void watched.exited.then(cleanup).catch(() => {});
			} catch { /* A retained private transcript is safer than a failed compaction. */ }
			options?.signal?.removeEventListener("abort", onAbort);
			this.controllers.delete(controller);
		}
		if (controller.signal.aborted) {
			response.stopReason = "aborted";
			response.content = [];
			response.errorMessage = "Summary aborted";
		} else if (source && (!environmentMatches(source) || !source.valid || this.captures.get(options!.sessionId!) !== source)) {
			response.stopReason = "error";
			response.content = [];
			response.errorMessage = "Cached summary source became stale";
		}
		if (response.stopReason === "stop" || response.stopReason === "length") stream.push({ type: "done", reason: response.stopReason, message: response });
		else stream.push({ type: "error", reason: response.stopReason as "error" | "aborted", error: response });
		stream.end();
	}
}
