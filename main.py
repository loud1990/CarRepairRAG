from dotenv import load_dotenv
import os
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import sys
import time
import threading
import argparse
from langchain.callbacks.base import BaseCallbackHandler
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

def _parse_retrieval_args():
    """
    Parse CLI args controlling retrieval mode and parameters.
    Defaults come from environment variables (loaded via .env).
    """
    parser = argparse.ArgumentParser(description="Car Repair RAG Chatbot")
    parser.add_argument(
        "--retrieval-mode",
        choices=["semantic", "keyword", "hybrid"],
        default=os.getenv("RETRIEVAL_MODE", "hybrid"),
        help="Retrieval mode: semantic (dense), keyword (sparse), or hybrid.")
    parser.add_argument(
        "--k",
        type=int,
        default=int(os.getenv("TOP_K", "3")),
        help="Number of documents to retrieve (top-k).")
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=float(os.getenv("DENSE_WEIGHT", "0.5")),
        help="Weight for dense scores in hybrid fusion.")
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=float(os.getenv("SPARSE_WEIGHT", "0.5")),
        help="Weight for sparse scores in hybrid fusion.")
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=int(os.getenv("RRF_K", "60")),
        help="Reciprocal Rank Fusion constant k.")
    return parser.parse_args()

load_dotenv()
llm = ChatOpenAI(model="gpt-5-nano", temperature=1, http_client=httpx.Client(verify=False), streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", http_client=httpx.Client(verify=False))
# Temporary SSL bypass; fix certificates with 'conda install ca-certificates certifi' for production.

# Configuration flags (can be set in .env)
use_hierarchical = os.getenv("HIERARCHICAL_CHUNKING", "true").lower() == "true"
rebuild_flag = os.getenv("REBUILD_VDB", "false").lower() == "true"

# Parse retrieval args (CLI with env defaults)
args = _parse_retrieval_args()

print("Welcome to the Car Repair RAG Chatbot! Type 'quit' to exit.")
print(f"Config: HIERARCHICAL_CHUNKING={use_hierarchical} | REBUILD_VDB={rebuild_flag}")
print(f"Retrieval Config: mode={args.retrieval_mode} | k={args.k} | dense_weight={args.dense_weight} | sparse_weight={args.sparse_weight} | rrf_k={args.rrf_k}")

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
)

while True:
    user_input = input("\nWhat is your question: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    try:
        spinner = Spinner("Thinking...")
        handler = StreamingWithSpinnerHandler(spinner, start_time=time.time(), min_delay=0.6, buffer_force_start_delay=0.5)
        # Ensure this handler is used by the LLM for this request:
        llm.callbacks = [handler]  # if llm is persistent
        spinner.start()

        result = run_agent(user_input, retriever=retriever, llm=llm)

        # If no tokens ever arrived, spinner might still be running:
        if spinner.is_running:
            spinner.stop_and_clear()
        print()  # newline to terminate streamed line
    except Exception as e:
        # Ensure spinner is cleared on any error path
        try:
            if 'spinner' in locals() and spinner.is_running:
                spinner.stop_and_clear()
        except Exception:
            pass
        print(f"Error: {e}")