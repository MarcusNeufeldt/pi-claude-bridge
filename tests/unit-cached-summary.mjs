import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, statSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { createSession, getSessionPath, openSession, deleteSession } from 'cc-session-io';
import { prepareSummaryFork, watchedSummaryProcess, CachedSummaries } from '../src/cached-summary.js';
const root=mkdtempSync(join(tmpdir(),'bridge-cached-unit-'));
process.env.CLAUDE_CONFIG_DIR=join(root,'claude');
const cwd=join(root,'cwd');mkdirSync(cwd);
process.on('exit',()=>rmSync(root,{recursive:true,force:true}));
const tool={name:'read',description:'Synthetic read',parameters:{type:'object',properties:{path:{type:'string'}}}};
const user=text=>({role:'user',content:text,timestamp:1});
const assistant=content=>({role:'assistant',content,provider:'claude-bridge',timestamp:2});
const text=s=>({type:'text',text:s});
const call=(id='tool_1')=>({type:'toolCall',id,name:'read',arguments:{path:'synthetic.txt'}});
const result=(id='tool_1')=>({role:'toolResult',toolCallId:id,content:[text('CEDAR_6842')],isError:false,timestamp:3});
function fixture({parallel=false}={}){
 const s=createSession({projectPath:cwd,claudeDir:process.env.CLAUDE_CONFIG_DIR});
 const first=s.addUserMessage('EARLIER_CEDAR');
 const calls=[call(),...(parallel?[call('tool_2')]:[])];
 const ids=[];
 for(const c of calls)ids.push(s.addAssistantMessage([{type:'tool_use',id:c.id,name:'mcp__custom-tools__read',input:c.arguments}]));
 const attachment=s.addAttachment({type:'file',filename:'evidence.txt',content:'NATIVE_ATTACHMENT'});
 s.save();
 const capture={sessionId:s.sessionId,cwd,context:{messages:[user('EARLIER_CEDAR')],tools:[tool]},tools:[tool],options:{cwd},valid:true};
 const history=[user('EARLIER_CEDAR'),assistant(calls),result(),...(parallel?[result('tool_2')]:[]),assistant([text('READY')]),user('UNSENT_A'),user('UNSENT_B')];
 return {s,capture,history,first,ids,attachment};
}

test('copies native UUIDs and attachments, appends all Pi-only results and consecutive user messages privately',()=>{
 const {s,capture,history,first,ids,attachment}=fixture();
 const before=readFileSync(s.jsonlPath),mtime=statSync(s.jsonlPath).mtimeMs;
 const fork=prepareSummaryFork(capture,history);
 const copy=openSession({sessionId:fork.sessionId,projectPath:cwd,claudeDir:process.env.CLAUDE_CONFIG_DIR});
 assert.notEqual(fork.sessionId,s.sessionId);assert.equal(fork.prefixAtoms,2);assert.equal(fork.suffixMessages,4);
 for(const id of [first,...ids,attachment])assert(copy.records.some(r=>r.uuid===id));
 assert(copy.records.every(r=>r.sessionId===fork.sessionId));
 const serialized=JSON.stringify(copy.records);
 for(const marker of ['EARLIER_CEDAR','NATIVE_ATTACHMENT','CEDAR_6842','READY','UNSENT_A','UNSENT_B'])assert(serialized.includes(marker));
 const response=copy.messages.find(m=>Array.isArray(m.message.content)&&m.message.content.some(b=>b.type==='tool_result'));
 assert.equal(response.message.content[0].tool_use_id,'tool_1');
 assert.deepEqual(readFileSync(s.jsonlPath),before);assert.equal(statSync(s.jsonlPath).mtimeMs,mtime);
 deleteSession(fork.sessionId,cwd,process.env.CLAUDE_CONFIG_DIR);
});

test('parallel native tool blocks remain paired with every Pi result',()=>{
 const {capture,history}=fixture({parallel:true});
 const fork=prepareSummaryFork(capture,history);
 const copy=openSession({sessionId:fork.sessionId,projectPath:cwd});
 const results=copy.messages.flatMap(m=>Array.isArray(m.message.content)?m.message.content:[]).filter(b=>b.type==='tool_result');
 assert.deepEqual(results.map(r=>r.tool_use_id),['tool_1','tool_2']);
});

test('idle native answer is included, unseen later native messages are not copied',()=>{
 const {s,capture,history}=fixture();
 s.addToolResults([{toolUseId:'tool_1',content:'CEDAR_6842'}]);s.addAssistantMessage([text('READY')]);
 const answerId=s.messages.at(-1).uuid;
 s.addUserMessage('NATIVE_FUTURE_NOT_IN_PI');s.save();
 const fork=prepareSummaryFork(capture,history.slice(0,4));
 const copy=openSession({sessionId:fork.sessionId,projectPath:cwd});
 assert(copy.messages.some(r=>r.uuid===answerId));assert(!JSON.stringify(copy.records).includes('NATIVE_FUTURE_NOT_IN_PI'));
 assert.equal(fork.suffixMessages,0);
});

test('parallel results with an interleaved steer decline rather than erase the steer or duplicate results',()=>{
 const {s,capture}=fixture({parallel:true});
 s.addToolResults([{toolUseId:'tool_1',content:'CEDAR_6842'},{toolUseId:'tool_2',content:'CEDAR_6842'}]);s.save();
 const history=[user('EARLIER_CEDAR'),assistant([call(),call('tool_2')]),result(),user('STEER_MUST_SURVIVE'),result('tool_2')];
 assert.throws(()=>prepareSummaryFork(capture,history),/reordering/);
});

test('native duration metadata is copied without becoming model content',()=>{
 const {s,capture,history}=fixture();
 const record={type:'system',subtype:'turn_duration',uuid:randomUUID(),parentUuid:s.records.at(-1).uuid,sessionId:s.sessionId,durationMs:100};
 writeFileSync(s.jsonlPath,readFileSync(s.jsonlPath,'utf8')+JSON.stringify(record)+'\n');
 const fork=prepareSummaryFork(capture,history);
 const copied=openSession({sessionId:fork.sessionId,projectPath:cwd});assert(copied.records.some(r=>r.uuid===record.uuid));
});

test('native signed thinking is preserved even when not needed for alignment',()=>{
 const {s,capture,history}=fixture();
 const records=s.records;
 records[1].message.content.unshift({type:'thinking',thinking:'NATIVE_THOUGHT',signature:'SIGNED_NATIVE'});
 writeFileSync(s.jsonlPath,records.map(r=>JSON.stringify(r)).join('\n')+'\n');
 const fork=prepareSummaryFork(capture,history);
 assert(readFileSync(getSessionPath(fork.sessionId,cwd),'utf8').includes('SIGNED_NATIVE'));
});

test('rejects incomplete calls, orphan results, truncated JSONL, missing parent and unprovable native prefix',()=>{
 for(const kind of ['pending','orphan','truncated','parent','prefix']){
  const {s,capture,history}=fixture();let selected=history;
  if(kind==='pending')selected=history.slice(0,2);
  if(kind==='orphan')selected=[history[0],result('alien')];
  if(kind==='truncated')writeFileSync(s.jsonlPath,readFileSync(s.jsonlPath,'utf8')+'{"type":');
  if(kind==='parent'){const r=s.records;r[1].parentUuid=randomUUID();writeFileSync(s.jsonlPath,r.map(x=>JSON.stringify(x)).join('\n')+'\n');}
  if(kind==='prefix')selected=[user('DIFFERENT'),...history.slice(1)];
  assert.throws(()=>prepareSummaryFork(capture,selected),undefined,kind);
 }
});

test('queued native steering is refused rather than silently included or omitted',()=>{
 const {s,capture,history}=fixture();s.addAttachment({type:'queued_command',command:'UNATTRIBUTED_STEER'});s.save();
 assert.throws(()=>prepareSummaryFork(capture,history),/Queued native steer/);
});

test('forks are independently owned and do not overwrite each other',()=>{
 const {capture,history}=fixture();
 const a=prepareSummaryFork(capture,history),b=prepareSummaryFork(capture,history);
 assert.notEqual(a.sessionId,b.sessionId);
 deleteSession(a.sessionId,cwd);assert(existsSync(getSessionPath(b.sessionId,cwd)));
});

test('capture clones only immutable request data and invalidation is session-scoped',()=>{
 const cache=new CachedSummaries(()=>{}),context={messages:[user('FIRST')],tools:[tool]};
 const a=cache.record('A',{id:'same'},context,{cwd,mcpServers:{live:{instance:{uncopyable:()=>{}}}}},[tool]);
 const b=cache.record('B',{id:'same'},context,{cwd},[tool]);assert(a&&b);
 context.messages[0].content='CHANGED';assert.equal(a.context.messages[0].content,'FIRST');assert.equal(a.options.mcpServers,undefined);
 cache.invalidate('A');assert.equal(a.valid,false);assert.equal(b.valid,true);
});

test('replacing a capture retires its predecessor',()=>{
 const cache=new CachedSummaries(()=>{}),context={messages:[user('FIRST')],tools:[tool]};
 const a=cache.record('A',{id:'same'},context,{cwd},[tool]);
 const b=cache.record('A',{id:'same'},context,{cwd},[tool]);
 assert.equal(a.valid,false);assert.equal(b.valid,true);
});

test('stderr flooding cannot block summary completion',async()=>{
 const watcher=watchedSummaryProcess();
 const child=watcher.start({command:process.execPath,args:['-e',"process.stderr.write('x'.repeat(16*1024*1024),()=>process.exit(0))"],cwd,env:process.env,signal:new AbortController().signal});
 assert.equal(await watcher.close(),true);assert.equal(child.exitCode,0);
});

test('process ownership refuses late spawn and observes actual close',async()=>{
 const closed=watchedSummaryProcess();assert.equal(await closed.close(),true);
 const options={command:process.execPath,args:['-e','setTimeout(()=>{},200)'],cwd,env:process.env,signal:new AbortController().signal};
 assert.throws(()=>closed.start(options),/cancelled/);
 const watcher=watchedSummaryProcess();const child=watcher.start(options);
 assert.equal(await watcher.close(),true);assert.notEqual(child.exitCode,null);
});
