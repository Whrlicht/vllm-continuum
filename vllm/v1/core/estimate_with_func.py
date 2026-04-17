from vllm.v1.request import Request
from typing import Optional
import os
import time
import re
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer
from vllm.v1.metrics.monitoring import monitoring_recorder

logger = init_logger(__name__)

FIXED_THRESHOLD_CONTINUUM = 2.0  # seconds

class Continuum_Recorder:
    def __init__(self):
        self.job_id_to_history = {}
        # Track scheduling operation timing
        self.scheduling_times = []  # List of {start_time, end_time, duration}
        self.output_dir: Optional[str] = None

    def set_output_dir(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def _get_output_dir(self) -> str:
        if self.output_dir:
            return self.output_dir
        return os.environ.get("RUN_OUTPUT_DIR", "./continuum_exp")

    def print_history(self):
        import json

        # Per-run output directory (set by launcher); fallback to default
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        # Atomic write to avoid partial reads by other processes
        final_path = os.path.join(output_dir, "scheduler_timestamps")
        tmp_path = f"{final_path}.tmp.{os.getpid()}.{time.time_ns()}"
        with open(tmp_path, "w") as f:
            json.dump(self.job_id_to_history, f, indent=2)
        os.replace(tmp_path, final_path)

    def request_arrives(self, request: Request):
        key = request.job_id if request.job_id is not None else request.request_id
        if key not in self.job_id_to_history:
            self.job_id_to_history[key] = []
        self.job_id_to_history[key].append({"Request_arrival_time": time.time()})
        monitoring_recorder.record_request_meta(request)
    
    def request_finished(self, request: Request):
        key = request.job_id if request.job_id is not None else request.request_id
        if key not in self.job_id_to_history:
            self.job_id_to_history[key] = []
        self.job_id_to_history[key].append({
            "Request_departure_time": time.time(),
            "num_generation_tokens": request.num_output_tokens,
        })

    def request_evicted_from_running_queue(self, request: Request):
        key = request.job_id if request.job_id is not None else request.request_id
        if key not in self.job_id_to_history:
            self.job_id_to_history[key] = []
        self.job_id_to_history[key].append({"Request_evicted_from_running_queue_time": time.time()})

    def request_pinned(self, request: Request):
        key = request.job_id if request.job_id is not None else request.request_id
        if key not in self.job_id_to_history:
            self.job_id_to_history[key] = []
        self.job_id_to_history[key].append({"pinned_time": time.time()})

    def request_unpinned(self, request: Request):
        key = request.job_id if request.job_id is not None else request.request_id
        if key not in self.job_id_to_history:
            self.job_id_to_history[key] = []
        self.job_id_to_history[key].append({"unpinned_time": time.time()})

    def request_waiting_to_running(self, request: Request, prompt_length: int, hit_length: int = 0):
        key = request.job_id if request.job_id is not None else request.request_id
        if key not in self.job_id_to_history:
            self.job_id_to_history[key] = []
        self.job_id_to_history[key].append({
            "waiting_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })
    
    def request_evicted_to_running(self, request: Request, prompt_length: int, hit_length: int):
        key = request.job_id if request.job_id is not None else request.request_id
        if key not in self.job_id_to_history:
            self.job_id_to_history[key] = []
        self.job_id_to_history[key].append({
            "evicted_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })

class ToolCallParser:
    """Parser for extracting function calls from LLM output.

    Uses the same parsing logic as mini-swe-agent to extract bash commands
    from markdown code blocks and identify the function call.

    This can be extended for other datasets with different parsing logic.
    """

    def parse(self, text: str) -> Optional[str]:
        """Parse LLM output and extract the function call name.

        Args:
            text: Output text from the LLM

        Returns:
            The function call name (e.g., "ls", "cd", "git"), or None if not found
        """
        # Same regex pattern as mini-swe-agent: r"```bash\s*\n(.*?)\n```"
        actions = re.findall(r"```bash\s*\n(.*?)\n```", text, re.DOTALL)

        if len(actions) == 1:
            bash_action = actions[0].strip()
            # Extract the first word (command) from the action
            words = bash_action.split()
            if words:
                return words[0]

        return None

class ToolCallEstimator:
    def __init__(
        self,
        tokenizer: Optional[AnyTokenizer] = None,
        model_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_revision: Optional[str] = None,
        parser: Optional[ToolCallParser] = None,
    ):
        self.func_call_to_exec_time: dict[str, float] = {}
        self.record_func_call_to_exec_time: dict[str, list[float]] = {}

        self.job_to_history: dict[str, list[dict[str, float]]] = {}

        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name is not None:
            try:
                self.tokenizer = get_tokenizer(
                    tokenizer_name=model_name,
                    tokenizer_mode=tokenizer_mode,
                    trust_remote_code=trust_remote_code,
                    revision=tokenizer_revision,
                )
                logger.info(f"Initialized tokenizer for model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer for {model_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Initialize parser (can be customized for different datasets)
        self.parser = parser if parser is not None else ToolCallParser()

    @staticmethod
    def _history_key(request: Request) -> str:
        # If job_id is not provided, isolate each request by request_id so
        # unrelated requests do not share estimator history.
        return request.job_id if request.job_id is not None else request.request_id

    def get_func_call_exec_time(self, func: str) -> Optional[float]:
        if func not in self.func_call_to_exec_time:
            return None
        return self.func_call_to_exec_time[func]
    
    #TODO Hanchen This is currently just an average 
    def update_func_call_exec_time(self, job_id: str) -> None:
        #this is called when the func call is back again in scheduler.py, update the exec time with last_func_call
        if job_id not in self.job_to_history or not self.job_to_history[job_id]:
            return

        last_entry = self.job_to_history[job_id][-1]
        if not isinstance(last_entry, dict):
            return

        last_departure_time = last_entry.get("departure_time")
        func = last_entry.get("func_call")
        if not isinstance(last_departure_time, (int, float)):
            return
        if not isinstance(func, str) or not func:
            return

        exec_time = time.time() - last_departure_time

        if func not in self.record_func_call_to_exec_time:
            self.record_func_call_to_exec_time[func] = [exec_time]
        else:
            self.record_func_call_to_exec_time[func].append(exec_time)
        self.func_call_to_exec_time[func] = sum(self.record_func_call_to_exec_time[func]) / len(self.record_func_call_to_exec_time[func])
        monitoring_recorder.record_tool_call_time(job_id, func, exec_time)
        return 
    
    #Functions below will be called by outside functions
    def set_up_pin(self, request: Request) -> float:
        if request.this_func_call is None:
            return 0
        
        this_func_call_exec_time = self.get_func_call_exec_time(request.this_func_call) or 0.0

        if this_func_call_exec_time > FIXED_THRESHOLD_CONTINUUM:
            return 0
        
        return FIXED_THRESHOLD_CONTINUUM

    def request_arrives(self, request: Request) -> None:
        logger.info(f"Request job id arriving: {request.job_id}, time is {time.time()}")
        # this is called when a job arrives in scheduler.py, if job is new, create an entry,
        history_key = self._history_key(request)
        if history_key not in self.job_to_history:
            self.job_to_history[history_key] = []
            assert request.last_func_call is None
            self.job_to_history[history_key].append(
                {"arrival_time": request.arrival_time})
            return

        history = self.job_to_history[history_key]
        request.last_func_call = None
        if history:
            last_entry = history[-1]
            if isinstance(last_entry, dict):
                last_func_call = last_entry.get("func_call")
                if isinstance(last_func_call, str):
                    request.last_func_call = last_func_call
        logger.info(f"Request job id: {request.job_id}, last func call: {request.last_func_call}")

        self.update_func_call_exec_time(history_key)

        self.job_to_history[history_key].append({"arrival_time": request.arrival_time})
        return
    
    def request_finished(self, request: Request) -> None:
        logger.info(f"Request job id finishing: {request.job_id}, time is {time.time()}")

        # Detokenize output and parse function call
        this_func_call = None
        if self.tokenizer is not None and len(request.output_token_ids) > 0:
            try:
                # Detokenize the output tokens
                output_text = self.tokenizer.decode(
                    request.output_token_ids,
                    skip_special_tokens=True
                )

                # Parse function call using the parser
                this_func_call = self.parser.parse(output_text)

                if this_func_call:
                    logger.info(f"Extracted func_call: {this_func_call} from output")
                else:
                    logger.debug(f"No function call found in output: {output_text[:200]}")
            except Exception as e:
                logger.warning(f"Error detokenizing/parsing output for request {request.request_id}: {e}")

        request.this_func_call = this_func_call
        history_key = self._history_key(request)
        if history_key not in self.job_to_history:
            self.job_to_history[history_key] = []
        self.job_to_history[history_key].append({
            "departure_time": time.time(),
            "func_call": request.this_func_call
        })
        return
