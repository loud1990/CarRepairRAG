from dotenv import load_dotenv
import os
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import sys
import time
import threading
import argparse
import random
import re
import unicodedata
from langchain.callbacks.base import BaseCallbackHandler
from langsmith import traceable
from langgraph.errors import GraphRecursionError
from src import DocumentLoader, VectorStoreManager
from src.agent import run_agent

class Spinner:
    def __init__(self, message="Thinking...", interval=0.1):
        self.message = message
        self.interval = 0.1 if interval is None else interval
        self._stop = threading.Event()
        self._thread = None
        self._last_len = 0
        self._frames = ["|", "/", "-", "\\"]

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self._frames[i % len(self._frames)]
            text = f"{self.message} {frame}"
            self._last_len = len(text)
            sys.stdout.write("\r" + text)
            sys.stdout.flush()
            time.sleep(self.interval)
            i += 1

    def stop_and_clear(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)
        # Clear the spinner line
        sys.stdout.write("\r" + " " * self._last_len + "\r")
        sys.stdout.flush()

    @property
    def is_running(self):
        return not self._stop.is_set()


class StreamingWithSpinnerHandler(BaseCallbackHandler):
    def __init__(
        self,
        spinner: Spinner,
        start_time: float | None = None,
        min_delay: float = 0.6,
        buffer_force_start_delay: float = 0.5,
    ):
        self.spinner = spinner
        self.start_time = start_time if start_time is not None else time.time()
        self.min_delay = float(min_delay)
        self.buffer_force_start_delay = float(buffer_force_start_delay)

        # Tool/phase tracking
        self.saw_tool = False
        self.tool_active = False
        self.tool_phase_complete = False

        # Streaming gate + buffering
        self._allowed_to_stream = False       # gate opened (post-tools or after min_delay)
        self._gate_allowed_time: float | None = None
        self._stream_started = False          # header printed and spinner cleared
        self._buffer: list[str] = []          # tokens buffered until we decide to show
        self._lock = threading.Lock()

    # Tool lifecycle detection
    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        with self._lock:
            self.saw_tool = True
            self.tool_active = True

    def on_tool_end(self, output, **kwargs) -> None:
        with self._lock:
            self.tool_active = False
            self.tool_phase_complete = True

    # Helper: visible token heuristic
    def _is_visible_token(self, token: str) -> bool:
        return bool(token) and (token.strip() != "")

    # Step 1: allow streaming once conditions are met (but don't clear spinner yet)
    def _allow_stream_if_ready(self) -> None:
        if self._allowed_to_stream:
            return
        now = time.time()
        if self.saw_tool:
            if self.tool_phase_complete and not self.tool_active:
                self._allowed_to_stream = True
                self._gate_allowed_time = now
        else:
            if (now - self.start_time) >= self.min_delay:
                self._allowed_to_stream = True
                self._gate_allowed_time = now

    # Step 2: if we can't wait for a visible token any longer, force-start and flush buffer
    def _force_start_if_buffer_timed_out(self) -> None:
        if self._stream_started or not self._allowed_to_stream:
            return
        if self._gate_allowed_time is None:
            return
        if (time.time() - self._gate_allowed_time) >= self.buffer_force_start_delay:
            try:
                self.spinner.stop_and_clear()
            except Exception:
                pass
            sys.stdout.write("Assistant: ")
            sys.stdout.flush()
            # Flush buffered tokens (whitespace or otherwise)
            if self._buffer:
                sys.stdout.write("".join(self._buffer))
                sys.stdout.flush()
                self._buffer.clear()
            self._stream_started = True

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        with self._lock:
            # Open the gate if conditions are satisfied
            self._allow_stream_if_ready()

            if not self._stream_started:
                if self._is_visible_token(token) and self._allowed_to_stream:
                    # First visible token after gate: clear spinner, print header, flush any buffered tokens, then print this token
                    try:
                        self.spinner.stop_and_clear()
                    except Exception:
                        pass
                    sys.stdout.write("Assistant: ")
                    sys.stdout.flush()
                    if self._buffer:
                        sys.stdout.write("".join(self._buffer))
                        self._buffer.clear()
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    self._stream_started = True
                    return
                else:
                    # Not yet visible or not yet allowed: buffer and possibly force-start after timeout
                    self._buffer.append(token)
                    self._force_start_if_buffer_timed_out()
                    return

            # Already streaming: print token directly
            sys.stdout.write(token)
            sys.stdout.flush()

    def on_llm_error(self, error, **kwargs) -> None:
        try:
            if self.spinner.is_running:
                self.spinner.stop_and_clear()
        except Exception:
            pass

class CandidateRetriever:
    """Shim to produce per-candidate context variations without changing agent code.
    It samples/shuffles a subset of docs from the base retriever for each candidate.
    """
    def __init__(self, base, sample_k: int | None, shuffle: bool, rng: random.Random):
        self.base = base
        self.sample_k = int(sample_k) if sample_k is not None else None
        self.shuffle = bool(shuffle)
        self.rng = rng

    def invoke(self, query: str):
        docs = self.base.invoke(query)
        try:
            n = len(docs)
        except Exception:
            return docs
        subset = list(docs)
        if self.sample_k is not None and self.sample_k < n:
            try:
                subset = self.rng.sample(subset, self.sample_k)
            except ValueError:
                subset = subset[: self.sample_k]
        if self.shuffle:
            try:
                self.rng.shuffle(subset)
            except Exception:
                pass
        return subset

def _normalize_for_vote(text: str) -> str:
    """Normalize answer strings for majority voting."""
    s = unicodedata.normalize("NFKC", str(text) if text is not None else "")
    s = s.lower().strip()
    # remove punctuation and control characters
    s = "".join(ch for ch in s if not (unicodedata.category(ch).startswith("P") or unicodedata.category(ch).startswith("C")))
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_retrieval_args():
    """
    Parse CLI args controlling retrieval mode and parameters, plus Self-Consistency & Multi-Query Fusion.
    Defaults come from environment variables (loaded via .env).
    """
    parser = argparse.ArgumentParser(description="Car Repair RAG Chatbot")
    parser.add_argument(
        "--retrieval-mode",
        choices=["semantic", "keyword", "hybrid"],
        default=os.getenv("RETRIEVAL_MODE", "hybrid"),
        help="Retrieval mode: semantic (dense), keyword (sparse), or hybrid.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=int(os.getenv("TOP_K", "3")),
        help="Number of documents to retrieve (top-k).",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=float(os.getenv("DENSE_WEIGHT", "0.5")),
        help="Weight for dense scores in hybrid fusion.",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=float(os.getenv("SPARSE_WEIGHT", "0.5")),
        help="Weight for sparse scores in hybrid fusion.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=int(os.getenv("RRF_K", "60")),
        help="Reciprocal Rank Fusion constant k.",
    )

    group = parser.add_argument_group("Self-Consistency & Multi-Query Fusion")
    group.add_argument(
        "--self-consistency",
        choices=["off", "majority"],
        default=os.getenv("SELF_CONSISTENCY", "off"),
        help="Self-consistency orchestration mode (off, majority).",
    )
    group.add_argument(
        "--sc-candidates",
        type=int,
        default=int(os.getenv("SC_CANDIDATES", "3")),
        help="Number of candidate answers K.",
    )
    dpc_env = os.getenv("SC_DOCS_PER_CANDIDATE")
    group.add_argument(
        "--sc-docs-per-candidate",
        type=int,
        default=(int(dpc_env) if dpc_env not in (None, "") else None),
        help="Subset of retrieved docs per candidate (<= k). If not provided, use k.",
    )
    group.add_argument(
        "--sc-shuffle-contexts",
        action="store_true",
        default=(os.getenv("SC_SHUFFLE_CONTEXTS", "").strip().lower() in ("1", "true", "yes")),
        help="If set, shuffle the per-candidate doc order before formatting.",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", "1")),
        help="Base LLM temperature for normal/SC runs.",
    )
    group.add_argument(
        "--sc-temperature-jitter",
        type=float,
        default=float(os.getenv("SC_TEMPERATURE_JITTER", "0.0")),
        help="Per-candidate jitter added/subtracted to base temperature.",
    )
    seed_env = os.getenv("SC_RANDOM_SEED")
    group.add_argument(
        "--sc-random-seed",
        type=int,
        default=(int(seed_env) if seed_env not in (None, "") else None),
        help="Seed for repeatable candidate sampling/shuffling.",
    )
    group.add_argument(
        "--multi-query",
        action="store_true",
        default=(os.getenv("MULTI_QUERY", "").strip().lower() in ("1", "true", "yes")),
        help="Enable multi-query expansion/fusion.",
    )
    group.add_argument(
        "--num-query-variants",
        type=int,
        default=int(os.getenv("NUM_QUERY_VARIANTS", "4")),
        help="Number of query variants for expansion (>= 2 when enabled).",
    )
    group.add_argument(
        "--expansion-method",
        choices=["heuristic", "llm"],
        default=os.getenv("EXPANSION_METHOD", "heuristic"),
        help="Expansion method for multi-query fusion.",
    )
    mqr_env = os.getenv("MULTI_QUERY_RRF_K")
    group.add_argument(
        "--multi-query-rrf-k",
        type=int,
        default=(int(mqr_env) if mqr_env not in (None, "") else None),
        help="Overrides RRF_K for multi-query fusion.",
    )
    return parser.parse_args()

load_dotenv()

# Parse CLI args (retrieval + self-consistency + multi-query fusion)
args = _parse_retrieval_args()

# LLM and embeddings using CLI/env temperature
llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "gpt-5-nano"),
    temperature=float(args.temperature),
    http_client=httpx.Client(verify=False),
    streaming=True,
)
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    http_client=httpx.Client(verify=False),
)
# Temporary SSL bypass; fix certificates with 'conda install ca-certificates certifi' for production.

# Configuration flags (can be set in .env)
use_hierarchical = os.getenv("HIERARCHICAL_CHUNKING", "true").lower() == "true"
rebuild_flag = os.getenv("REBUILD_VDB", "false").lower() == "true"

print("Welcome to the Car Repair RAG Chatbot! Type 'quit' to exit.")
print(f"Config: HIERARCHICAL_CHUNKING={use_hierarchical} | REBUILD_VDB={rebuild_flag}")
print(f"Retrieval Config: mode={args.retrieval_mode} | k={args.k} | dense_weight={args.dense_weight} | sparse_weight={args.sparse_weight} | rrf_k={args.rrf_k}")
print(f"SC/MQ Config: self_consistency={args.self_consistency} | sc_candidates={args.sc_candidates} | sc_docs_per_cand={args.sc_docs_per_candidate} | sc_shuffle={args.sc_shuffle_contexts} | temperature={args.temperature} | sc_jitter={args.sc_temperature_jitter} | seed={args.sc_random_seed} | multi_query={args.multi_query} | num_variants={args.num_query_variants} | expansion_method={args.expansion_method} | multi_query_rrf_k={args.multi_query_rrf_k}")

manager = VectorStoreManager(embeddings)
if rebuild_flag:
    manager.rebuild()
vectorstore = manager.get_or_create()
retriever = manager.build_retriever(
    mode=args.retrieval_mode,
    k=args.k,
    dense_weight=args.dense_weight,
    sparse_weight=args.sparse_weight,
    rrf_k=args.rrf_k,
    multi_query=args.multi_query,
    num_query_variants=args.num_query_variants,
    expansion_method=args.expansion_method,
    expansion_llm=(llm if args.expansion_method == "llm" else None),
    multi_query_rrf_k=args.multi_query_rrf_k,
)

@traceable(name="query_handler", metadata={"retrieval_mode": args.retrieval_mode, "self_consistency": args.self_consistency})
def process_query(user_input: str, retriever, llm, args, handler):
    """Process user query with optional self-consistency."""
    use_sc = (args.self_consistency == "majority") and (int(args.sc_candidates or 0) > 1)
    
    if not use_sc:
        # Normal path: preserve existing streaming behavior
        llm.callbacks = [handler]
        result = run_agent(user_input, retriever=retriever, llm=llm)
        return result, None
    else:
        # Self-consistency (majority voting) path
        base_model = getattr(llm, "model", os.getenv("CHAT_MODEL", "gpt-5-nano"))
        rng_seed = args.sc_random_seed
        base_rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        candidates: list[str] = []
        counts: dict[str, int] = {}
        norm_map: dict[str, dict] = {}
        sample_k = args.sc_docs_per_candidate if args.sc_docs_per_candidate is not None else args.k
        try:
            sample_k = int(sample_k)
        except Exception:
            sample_k = args.k

        for i in range(int(args.sc_candidates)):
            jitter_span = float(args.sc_temperature_jitter)
            jitter = base_rng.uniform(-jitter_span, jitter_span) if jitter_span > 0 else 0.0
            temp_i = max(0.0, min(2.0, float(args.temperature) + float(jitter)))
            llm_i = ChatOpenAI(
                model=base_model,
                temperature=temp_i,
                http_client=httpx.Client(verify=False),
                streaming=False,
            )
            cand_retriever = CandidateRetriever(
                base=retriever,
                sample_k=sample_k,
                shuffle=bool(args.sc_shuffle_contexts),
                rng=random.Random((int(rng_seed) if rng_seed is not None else 0) + i),
            )
            ans_i = run_agent(user_input, retriever=cand_retriever, llm=llm_i)
            candidates.append(ans_i)

        # Aggregate votes
        for idx, ans in enumerate(candidates):
            key = _normalize_for_vote(ans)
            if key not in norm_map:
                norm_map[key] = {"text": ans, "first_index": idx}
            counts[key] = counts.get(key, 0) + 1

        def _select_winner_key(items):
            return max(
                items,
                key=lambda kv: (kv[1], len(norm_map[kv[0]]["text"]), -norm_map[kv[0]]["first_index"]),
            )[0]

        winner_key = _select_winner_key(list(counts.items())) if counts else ""
        final_answer = norm_map.get(winner_key, {"text": (candidates[0] if candidates else "")})["text"]
        
        return final_answer, counts

while True:
    user_input = input("\nWhat is your question: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    try:
        spinner = Spinner("Thinking...")
        handler = StreamingWithSpinnerHandler(spinner, start_time=time.time(), min_delay=0.6, buffer_force_start_delay=0.5)
        spinner.start()

        result, counts = process_query(user_input, retriever, llm, args, handler)
        
        # If no tokens ever arrived, spinner might still be running:
        if spinner.is_running:
            spinner.stop_and_clear()
        
        if counts is not None:
            # Self-consistency path output
            print("Self-Consistency summary (majority vote):")
            for k, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                snippet = k[:100]
                print(f"- {c}x: {snippet}")
            print("\nAssistant: " + result)
        else:
            # Normal path output
            print()  # newline to terminate streamed line

    except GraphRecursionError as e:
        # Handle LangGraph recursion limit error with user-friendly message
        try:
            if 'spinner' in locals() and spinner.is_running:
                spinner.stop_and_clear()
        except Exception:
            pass
        print(f"\n⚠️  Error: The system encountered a recursion limit while processing your question.")
        print(f"\n{str(e)}")
        print(f"\nThe chatbot is still running and ready for your next question.")
    except Exception as e:
        # Ensure spinner is cleared on any error path
        try:
            if 'spinner' in locals() and spinner.is_running:
                spinner.stop_and_clear()
        except Exception:
            pass
        print(f"\n⚠️  Error: {e}")