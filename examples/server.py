#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server on exllamav3 (llama-server's CLI shape).

  server.py -m <exl3 dir> --host 127.0.0.1 --port N [-np 1] [--no-warmup] [-c CTX] [--no-mtp] [other args ignored]

GET /health; POST /v1/chat/completions with messages, max_tokens, temperature, top_p, top_k, min_p, presence_penalty,
frequency_penalty, repetition_penalty, dry_multiplier/dry_base/dry_allowed_length/dry_penalty_last_n (llama.cpp names),
seed, chat_template_kwargs (enable_thinking, reasoning_effort ...), stop_on_loop (default on: exllamav3's loop detector
ends a run that repeats itself, finish_reason "loop"), thinking_budget (chat only, never benchmarks: at N generated tokens with the thinking still open,
the end of thinking is injected and the model answers). One request at a time (like -np 1), MTP drafting
(--draft tokens) unless --no-mtp. Reasoning is split out as reasoning_content at the first </think>, like llama-server
--reasoning-format deepseek. Qwen XML tool calls are translated to OpenAI tool_calls. With stream=true,
reasoning and text are emitted incrementally. --speed-footer appends a decode-rate footer to natural-language replies (chat
UIs only: it is not model output, so benchmarks and APIs never get it; llama-server has no such footer).
"""
import argparse, json, os, re, sys, threading, time, traceback, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from exllamav3 import Generator, Job, model_init, ComboSampler

ap = argparse.ArgumentParser(); ap.add_argument('-m', required=True); ap.add_argument('--host', default='127.0.0.1')
ap.add_argument('--port', type=int, required=True); ap.add_argument('-c', type=int, default=65536); ap.add_argument('--no-mtp', action='store_true')
ap.add_argument('--chunk', default='1024', help='prefill chunk (tokens); smaller = less activation VRAM, same output')
ap.add_argument('--gs', default='auto', help='VRAM budget, GiB (exllamav3 -gs); auto = as chat.py and fit_check.py load')
ap.add_argument('--lora', default=os.environ.get('EXL3_LORA', ''), help='PEFT adapter dir loaded on the main model')
ap.add_argument('-cq', default='8', help='KV cache bits (exllamav3 -cq); 8 = near-lossless, matches the GGUF q8_0 KV')
# keep only our own flags: argparse would read llama-server flags such as '-ctk q8_0' as '-c tk'
ap.add_argument('--reasoning-effort', default='', help='default chat_template reasoning_effort (xhigh/medium/low) when a request sets none')
ap.add_argument('--api-key', default='', help='require Authorization: Bearer <key> (llama-server --api-key); /health stays open')
ap.add_argument('--speed-footer', action='store_true', help='append "*N tok/s · M tokens*" to natural-language replies (chat UI only)')
ap.add_argument('--draft', type=int, default=5, help='MTP draft tokens per round (max with --dynamic-draft)')
ap.add_argument('--dynamic-draft', action='store_true', help='cut each MTP draft where the calibrated acceptance estimate '
                'falls below --draft-confidence (exllamav3 dynamic_draft_tokens): fewer wasted drafts, higher acceptance')
ap.add_argument('--draft-confidence', type=float, default=0.4, help='target acceptance probability for --dynamic-draft')
ap.add_argument('--draft-p-min', type=float, default=0.0, help='MTP: draft a token only while the drafter gives it at least this '
                'probability at the request\'s temperature (llama.cpp --spec-draft-p-min); combines with --dynamic-draft')
ap.add_argument('--sparse-head', type=int, default=0, help='MTP drafts through a sparse LM head of this many frequent vocabulary '
                'columns (exllamav3 sparse_draft_head; verification keeps the full head, output unchanged)')
ap.add_argument('--spec-sampling', action='store_true', help='MTP drafts sampled from the request\'s own sampling distribution and '
                'accepted with probability min(1, p/q) (exllamav3 spec_sampling): same output distribution, more drafts kept')
ap.add_argument('--draft-temp', type=float, default=1.0, help='with --spec-sampling: drafter temperature as a multiple of the request\'s')
ap.add_argument('-np', type=int, default=1, help='parallel slots (llama-server -np): -c is the total cache, each request gets -c / np; '
                'requests are batched by the generator')
KNOWN = {'--draft': 1, '--sparse-head': 1, '--draft-p-min': 1, '--dynamic-draft': 0, '--draft-confidence': 1, '--spec-sampling': 0, '--draft-temp': 1, '-np': 1, '--api-key': 1, '--reasoning-effort': 1, '-m': 1, '--host': 1, '--port': 1, '-c': 1, '-cq': 1, '--chunk': 1, '--gs': 1, '--no-mtp': 0, '--lora': 1, '--speed-footer': 0}
argv, i = [], 1
while i < len(sys.argv):
    n = KNOWN.get(sys.argv[i])
    if n is None: i += 1; continue
    argv += sys.argv[i:i + 1 + n]; i += 1 + n
a = ap.parse_args(argv)   # repeated flags: the last wins (config extra comes after _base)
p = argparse.ArgumentParser(); model_init.add_args(p, cache=True, add_draft_model_args=True)
margs = p.parse_args(['-m', a.m, '-cs', str(a.c), '-cq', a.cq, '-chunk_size', a.chunk, '-gs', a.gs, '-ndt', str(a.draft)] + ([] if a.no_mtp else ['-mtp']))
torch.set_grad_enabled(False)
model, config, cache, tok, dmodel, dconfig, dcache = model_init.init(margs)
# Hydrate the HF chat-template tokenizer before accepting concurrent HTTP calls.
# A client can send a title request and a chat request at the same time; leaving
# this lazy lets both threads import transformers while it is still initializing.
from transformers import AutoTokenizer
tok.hf_tokenizer = AutoTokenizer.from_pretrained(a.m, local_files_only=True)
if a.lora:
    from exllamav3.model.lora import LoRA
    LoRA.from_directory(model, a.lora)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dmodel, draft_cache=dcache, num_draft_tokens=a.draft if dmodel else None,
                dynamic_draft_tokens=a.dynamic_draft, draft_confidence=a.draft_confidence,
                spec_sampling=a.spec_sampling, draft_temperature=a.draft_temp, draft_p_min=a.draft_p_min, sparse_draft_head=a.sparse_head)
SLOT_CTX = a.c // max(1, a.np)   # per-request context, as llama-server splits -c across -np slots
_glock = threading.Lock()          # held by the scheduler while it steps the generator
_pending = []; _routes = {}; _wake = threading.Condition()


def _scheduler():
    """One thread owns the generator: it enqueues new jobs between steps and routes every result to its request, so
    concurrent requests decode together (continuous batching) instead of queueing behind a lock."""
    import queue as _q   # noqa: F401
    with torch.inference_mode():
        while True:
            with _wake:
                while not _pending and not gen.num_remaining_jobs():
                    _wake.wait()
                new = list(_pending); _pending.clear()
            with _glock:
                for job, q in new:
                    gen.enqueue(job); _routes[id(job)] = q
                results = gen.iterate() if gen.num_remaining_jobs() else []
            for r in results:
                q = _routes.get(id(r.get('job')))
                if q is not None:
                    q.put(r)
                    if r.get('eos'):
                        _routes.pop(id(r.get('job')), None)


threading.Thread(target=_scheduler, daemon=True).start()
print(f'exl3_server ready on {a.host}:{a.port} model {a.m} ctx {a.c} mtp {not a.no_mtp}' + (f' lora {a.lora}' if a.lora else ''), flush=True)


def parse_tool_calls(content):
    """Convert Qwen's XML-like tool syntax to OpenAI function calls."""
    calls, spans = [], []
    pattern = re.compile(r'<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>', re.S)
    for match in pattern.finditer(content):
        args = {}
        for param in re.finditer(r'<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>', match.group(2), re.S):
            raw = param.group(2).strip()
            try:
                value = json.loads(raw)
            except (ValueError, TypeError):
                value = raw
            args[param.group(1)] = value
        calls.append(dict(id='call_' + uuid.uuid4().hex, type='function',
                          function=dict(name=match.group(1), arguments=json.dumps(args, ensure_ascii=False))))
        spans.append(match.span())
    for start, end in reversed(spans):
        content = content[:start] + content[end:]
    return calls, content.strip()


class StreamEmitter:
    """Split generated text into live reasoning/content SSE and Qwen tool calls."""
    THINK_END = '</think>'
    THINK_OPEN = '<think>'
    TOOL_OPEN = '<tool_call>'
    TOOL_CLOSE = '</tool_call>'

    def __init__(self, emit, thinking):
        self.emit = emit
        self.mode = 'reasoning' if thinking else 'content'
        self.pending = ''
        self.leading = self.mode == 'reasoning'
        self.tool_buffer = None
        self.tool_index = 0

    def _delta(self, key, value):
        if value:
            self.emit({key: value})

    def _tool_delta(self, raw):
        calls, remainder = parse_tool_calls(raw)
        if not calls:
            self._delta('content', raw)
            return
        for call in calls:
            self.emit({'tool_calls': [dict(index=self.tool_index, **call)]})
            self.tool_index += 1
        self._delta('content', remainder)

    def feed(self, fragment):
        if self.tool_buffer is not None:
            self.tool_buffer += fragment
            while self.TOOL_CLOSE in self.tool_buffer:
                end = self.tool_buffer.index(self.TOOL_CLOSE) + len(self.TOOL_CLOSE)
                complete, self.tool_buffer = self.tool_buffer[:end], self.tool_buffer[end:]
                self._tool_delta(complete)
                if not self.tool_buffer:
                    self.tool_buffer = None
                    break
                if self.TOOL_OPEN not in self.tool_buffer:
                    self._delta('content', self.tool_buffer)
                    self.tool_buffer = None
                    break
                start = self.tool_buffer.index(self.TOOL_OPEN)
                self._delta('content', self.tool_buffer[:start])
                self.tool_buffer = self.tool_buffer[start:]
            return
        self.pending += fragment
        self._drain(final=False)

    def _drain(self, final):
        while self.pending:
            if self.mode == 'reasoning':
                if self.leading:
                    if self.pending.startswith(self.THINK_OPEN):
                        self.pending = self.pending[len(self.THINK_OPEN):]
                        self.leading = False
                    elif self.THINK_OPEN.startswith(self.pending) and not final:
                        return
                    else:
                        self.leading = False
                index = self.pending.find(self.THINK_END)
                if index >= 0:
                    self._delta('reasoning_content', self.pending[:index])
                    self.pending = self.pending[index + len(self.THINK_END):]
                    self.mode = 'content'
                    continue
                safe = len(self.pending) if final else max(0, len(self.pending) - len(self.THINK_END) + 1)
                if safe:
                    self._delta('reasoning_content', self.pending[:safe])
                    self.pending = self.pending[safe:]
                return

            index = self.pending.find(self.TOOL_OPEN)
            if index >= 0:
                self._delta('content', self.pending[:index])
                self.tool_buffer = self.pending[index:]
                self.pending = ''
                if self.TOOL_CLOSE in self.tool_buffer:
                    self.feed('')
                return
            safe = len(self.pending) if final else max(0, len(self.pending) - len(self.TOOL_OPEN) + 1)
            if safe:
                self._delta('content', self.pending[:safe])
                self.pending = self.pending[safe:]
            return

    def finish(self):
        if self.tool_buffer is not None:
            if self.TOOL_CLOSE in self.tool_buffer:
                self.feed('')
            else:
                self._delta('content', self.tool_buffer)
                self.tool_buffer = None
        self._drain(final=True)


def complete(body, on_text=None):
    kw = dict(body.get('chat_template_kwargs') or {})
    if a.reasoning_effort and kw.get('enable_thinking', True): kw.setdefault('reasoning_effort', body.get('reasoning_effort') or a.reasoning_effort)
    elif body.get('reasoning_effort'): kw.setdefault('reasoning_effort', body['reasoning_effort'])
    messages = []
    for original in body['messages']:
        message = dict(original)
        calls = message.get('tool_calls')
        if isinstance(calls, list):
            normalized = []
            for original_call in calls:
                call = dict(original_call)
                function = dict(call.get('function') or {})
                arguments = function.get('arguments')
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        arguments = {'raw': arguments}
                if arguments is not None and not isinstance(arguments, dict):
                    arguments = {'value': arguments}
                if arguments is not None:
                    function['arguments'] = arguments
                call['function'] = function
                normalized.append(call)
            message['tool_calls'] = normalized
        messages.append(message)
    tools = body.get('tools')
    if body.get('tool_choice') == 'none' or not isinstance(tools, list) or not tools:
        tools = None
    ids = tok.hf_chat_template(messages=messages, add_generation_prompt=True, tools=tools, **kw)
    room = SLOT_CTX - ids.shape[-1]
    requested_new = max(1, int(body.get('max_tokens') or 4096))
    if room < min(64, requested_new):
        raise ValueError(f"prompt uses {ids.shape[-1]} tokens in a {SLOT_CTX}-token context, "
                         f"leaving only {room}; compact earlier turns or start a fresh chat")
    max_new = min(requested_new, room)
    t = float(body.get('temperature', 1.0))
    sampler = ComboSampler(temperature=t if t > 0 else 1.0, top_p=float(body.get('top_p', 0.95)), top_k=1 if t == 0 else int(body.get('top_k', 20)),
                           min_p=float(body.get('min_p', 0.0)), pres_p=float(body.get('presence_penalty', 0.0)),
                           freq_p=float(body.get('frequency_penalty', 0.0)), rep_p=float(body.get('repetition_penalty', 1.0)),
                           dry_multiplier=float(body.get('dry_multiplier', 0.0)), dry_base=float(body.get('dry_base', 1.75)),
                           dry_allowed_length=int(body.get('dry_allowed_length', 2)), dry_range=max(0, int(body.get('dry_penalty_last_n', 0))))
    job = Job(input_ids=ids, max_new_tokens=max_new, stop_conditions=config.eos_token_id_list, sampler=sampler, seed=int(body.get('seed', 1)),
              stop_on_loop=bool(body.get('stop_on_loop', True)))
    budget = int(body.get('thinking_budget') or 0)
    text, last, t0, n_gen, forced = '', None, time.time(), 0, False
    import queue
    q = queue.Queue()
    with _wake:
        _pending.append((job, q)); _wake.notify()
    while True:
        r = q.get()
        fragment = r.get('text', '')
        text += fragment
        if on_text and fragment:
            on_text(fragment)
        ti = r.get('token_ids'); n_gen += ti.shape[-1] if ti is not None else 0
        if r.get('eos'):
            last = r; break
        if budget and not forced and n_gen >= budget and '</think>' not in text:
            with _glock:
                job.constrain_output_now('\n\nThinking budget reached; answering now.\n</think>\n\n')
            forced = True
    if last and last.get('stage') == 'error':   # the generator reaps a failed job (e.g. OOM) into an error result
        err = last.get('error'); raise err if isinstance(err, BaseException) else RuntimeError(f'generation failed: {err!r}')
    er = (last or {}).get('eos_reason')
    finish = 'stop' if er == 'stop_token' else 'loop' if er == 'loop_detected' else 'length'
    head = tok.hf_chat_template(messages=[{'role': 'user', 'content': 'x'}], add_generation_prompt=True, **kw)
    prompt_tail = tok.decode(head[0, -4:]) if head.shape[-1] >= 4 else ''
    reasoning, content = '', text
    if '</think>' in text:
        reasoning, content = text.split('</think>', 1)
        reasoning = reasoning.removeprefix('<think>').strip(); content = content.lstrip()
    elif kw.get('enable_thinking', True) and '<think>' in prompt_tail:   # thinking never closed: all reasoning
        reasoning, content = text, ''
    tool_calls, content = parse_tool_calls(content)
    if tool_calls:
        finish = 'tool_calls'
    n = last['new_tokens'] if last else 0
    acc, rej = (last or {}).get('accepted_draft_tokens'), (last or {}).get('rejected_draft_tokens')
    generation_s = max(1e-9, float((last or {}).get('time_generate', time.time() - t0)))
    # llama-server's fields: prompt_n = tokens actually prefilled (cache hits excluded), cache_n = reused tokens
    cache_n = int((last or {}).get('cached_tokens') or 0); prefill_s = float((last or {}).get('time_prefill') or 0)
    prompt_n = max(0, int(ids.shape[-1]) - cache_n)
    timings = dict(cache_n=cache_n, prompt_n=prompt_n, prompt_ms=prefill_s * 1000,
                   prompt_per_second=prompt_n / prefill_s if prefill_s > 0 else None, predicted_n=n,
                   predicted_per_second=n / generation_s, predicted_ms=generation_s * 1000,
                   draft_n=(acc or 0) + (rej or 0) if acc is not None else None,
                   draft_n_accepted=acc)
    if forced: print(f'thinking budget {budget} reached: answer forced', flush=True)
    message = dict(role='assistant', content=content, reasoning_content=reasoning)
    if tool_calls:
        message['tool_calls'] = tool_calls
    return dict(id='exl3', object='chat.completion', model=a.m,
                choices=[dict(index=0, finish_reason=finish, message=message)],
                usage=dict(prompt_tokens=int(ids.shape[-1]), completion_tokens=n, total_tokens=int(ids.shape[-1]) + n),
                timings=timings)


def speed_footer(body, result):
    """Visible per-message decode speed (--speed-footer only), omitted when it would break strict output formats."""
    if not a.speed_footer:
        return ''
    choice = result['choices'][0]
    message = choice['message']
    timing = result.get('timings') or {}
    text = message.get('content') or ''
    if (choice.get('finish_reason') != 'stop' or not text or message.get('tool_calls') or not timing.get('predicted_n')
            or body.get('response_format')):
        return ''
    if re.search(r'```|</?(?:html|head|body|script|style|svg|canvas)\b', text, re.I):
        return ''
    user_text = next((str(m.get('content') or '') for m in reversed(body.get('messages') or [])
                      if m.get('role') == 'user'), '')
    if (re.search(r'\b(?:exactly|return only|output only|respond only|valid json|json object)\b',
                  user_text, re.I)
            or text.lstrip().startswith(('```', '{', '['))):
        return ''
    return f"\n\n*{float(timing['predicted_per_second']):.1f} tok/s · {int(timing['predicted_n'])} tokens*"


class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def begin_stream(self):
        self._stream_id = 'chatcmpl-' + uuid.uuid4().hex
        self._stream_created = int(time.time())
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache, no-transform')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        self._write_event({'id': self._stream_id, 'object': 'chat.completion.chunk',
                           'created': self._stream_created, 'model': 'qwen38-27b-exl3',
                           'choices': [{'index': 0, 'delta': {'role': 'assistant'},
                                        'finish_reason': None}]})

    def _write_event(self, obj):
        self.wfile.write(b'data: ' + json.dumps(obj, ensure_ascii=False).encode() + b'\n\n')
        self.wfile.flush()

    def send_delta(self, delta):
        if not delta:
            return
        if delta.get('tool_calls'):
            for i, call in enumerate(delta['tool_calls']):
                call.setdefault('index', i)
        self._write_event({'id': self._stream_id, 'object': 'chat.completion.chunk',
                           'created': self._stream_created, 'model': 'qwen38-27b-exl3',
                           'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})

    def end_stream(self, obj):
        choice = obj['choices'][0]
        timing = obj.get('timings') or {}
        telemetry = {k: timing.get(k) for k in ('predicted_n', 'predicted_ms',
                                                  'predicted_per_second', 'draft_n',
                                                  'draft_n_accepted')}
        final = {'id': self._stream_id, 'object': 'chat.completion.chunk',
                 'created': self._stream_created, 'model': 'qwen38-27b-exl3',
                 'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice['finish_reason']}],
                 'timings': timing, 'exl3_telemetry': telemetry}
        if self._include_usage:
            final['usage'] = obj['usage']
        self._write_event(final)
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()
    def authorized(self):
        if not a.api_key or self.headers.get('Authorization', '') == f'Bearer {a.api_key}':
            return True
        self.send(401, {'error': {'message': 'Invalid API Key', 'type': 'authentication_error'}}); return False
    def do_GET(self):
        if self.path.startswith('/health'):
            return self.send(200, {'status': 'ok'})
        if not self.authorized(): return
        if self.path == '/v1/models':
            mid = os.path.basename(os.path.normpath(a.m))
            return self.send(200, {'object': 'list', 'data': [dict(
                id=mid, object='model', created=int(time.time()), owned_by='local', root=mid)]})
        self.send(404, {'error': 'not found'})
    def do_POST(self):
        if not self.authorized(): return
        if not self.path.startswith('/v1/chat/completions'): return self.send(404, {'error': 'not found'})
        streaming = False
        try:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self._include_usage = bool((body.get('stream_options') or {}).get('include_usage'))
            if body.get('stream'):
                streaming = True
                self.begin_stream()
                template_kwargs = body.get('chat_template_kwargs') or {}
                emitter = StreamEmitter(
                    self.send_delta,
                    thinking=template_kwargs.get('enable_thinking', True),
                )
                result = complete(body, on_text=emitter.feed)
                emitter.finish()
                footer = speed_footer(body, result)
                if footer:
                    self.send_delta({'content': footer})
                self.end_stream(result)
            else:
                result = complete(body)
                footer = speed_footer(body, result)
                if footer:
                    result['choices'][0]['message']['content'] += footer
                self.send(200, result)
        except ValueError as e:
            error = {'error': {'message': str(e), 'type': 'invalid_request_error',
                               'code': 'context_length_exceeded'}}
            if streaming:
                try:
                    self._write_event(error)
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send(400, error)
        except Exception as e:
            traceback.print_exc(); sys.stderr.flush()   # the error goes to the server log
            if streaming:
                try:
                    self._write_event({'error': {'message': repr(e)}})
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send(500, {'error': {'message': repr(e)}})


ThreadingHTTPServer((a.host, a.port), H).serve_forever()
