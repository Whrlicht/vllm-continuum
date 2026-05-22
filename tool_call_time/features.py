"""特征提取: A 类 (命令结构) + B 类 (子参数语义) + E 类 (轨迹历史观测)。

A+B 是从单个 tool_call 直接抽出的 static 特征。
E 类需要按 trajectory 顺序滚动维护状态。

设计要点:
  - 训练 / 推理使用同一份提取逻辑, 完全对称。
  - E 类特征严格 causal: 计算第 j 个 call 的特征时, 只用 j 之前已观测到的真值。
  - 不依赖任何离散身份 ID (instance_id / repo / model name 都不进特征)。
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any, Optional

from bucket import classify, _parse_args, _strip_cd_prefix, _strip_chmod_prefix

# ---------------------------------------------------------------------------
# A + B: 静态特征
# ---------------------------------------------------------------------------

# 已知 "compile-heavy" 的 PyPI 包名 (PyPI 公共事实, 跨数据集稳定)
_COMPILE_HEAVY_PKGS = {
    'numpy', 'scipy', 'pandas', 'pillow', 'lxml', 'pydantic-core',
    'pydantic_core', 'cython', 'pyarrow', 'tensorflow', 'torch',
    'opencv-python', 'matplotlib', 'cffi', 'cryptography',
}

# "可能 hang" 的脚本内容关键词 (出现在文件名 / 命令中也算弱信号)
_HANG_HINT_TOKENS = ('reproduce', 'server', 'app', 'tornado', 'flask',
                     'bottle', 'uvicorn')


def _extract_pip_features(body: str) -> dict:
    out = {}
    m = re.search(r'pip\s+install\s+([^&|;]+)', body)
    target = m.group(1).strip() if m else ''
    out['pip_is_editable'] = int(bool(re.search(r'(^|\s)-e\b', target)))
    out['pip_has_extras'] = int(bool(re.search(r'\[[a-zA-Z,_-]+\]', target)))
    out['pip_no_build_iso'] = int('--no-build-isolation' in target)
    out['pip_no_deps'] = int('--no-deps' in target)
    out['pip_has_venv_chain'] = int(bool(
        re.search(r'(python\s*-m\s*venv|virtualenv)', body)))
    out['pip_has_uninstall'] = int('pip uninstall' in body)
    # 包名匹配 compile-heavy 列表
    pkgs = re.findall(r'(?:^|\s)([a-z][a-z0-9_\-]+)(?:==[^\s]+)?', target)
    out['pip_n_pkgs'] = len([p for p in pkgs if not p.startswith('-')])
    out['pip_has_compile_heavy'] = int(any(
        p.lower().replace('_', '-') in _COMPILE_HEAVY_PKGS for p in pkgs))
    return out


def _extract_pytest_features(body: str) -> dict:
    out = {}
    out['pytest_has_k'] = int(bool(re.search(r'-k\s', body)))
    out['pytest_has_x'] = int(bool(re.search(r'(?:^|\s)-x(?:\s|$)', body)))
    out['pytest_collect_only'] = int('--collect-only' in body)
    out['pytest_has_tb'] = int('--tb' in body)
    # path scope: 0=none/full, 1=dir, 2=file, 3=test_id
    after_pytest = re.search(r'pytest\b\s*([^&|;]*)', body)
    args_after = after_pytest.group(1).strip() if after_pytest else ''
    args_clean = re.sub(r'-\S+(?:\s+\S+)?', '', args_after).strip()
    if not args_clean:
        scope = 0
    elif '::' in args_clean:
        scope = 3
    elif args_clean.endswith('.py') or '.py ' in args_clean:
        scope = 2
    else:
        scope = 1
    out['pytest_scope'] = scope
    return out


def _extract_static_features(tool_call: dict) -> dict:
    """A + B 类特征。"""
    fn = tool_call.get('function') or {}
    name = fn.get('name', '')
    args = _parse_args(fn.get('arguments'))

    bucket = classify(tool_call)
    feats: dict[str, Any] = {
        'bucket': bucket,
        'tool_name': name,
    }

    if name == 'bash':
        c = (args.get('command') or '').strip()
        feats['cmd_len_chars'] = len(c)
        feats['cmd_len_tokens'] = len(c.split())
        feats['cmd_has_amp'] = int('&&' in c)
        feats['cmd_has_pipe'] = int(bool(re.search(r'(?<!\\)\|(?!\|)', c)))
        feats['cmd_has_bg'] = int(bool(re.search(r'(?<!&)&(?:\s|$)', c)))
        feats['cmd_has_redirect'] = int(bool(re.search(r'>\s|2>&1', c)))
        feats['cmd_amp_chain_len'] = c.count('&&')
        feats['cmd_n_pipes'] = len(re.findall(r'(?<!\\)\|(?!\|)', c))
        # timeout N 前缀
        m_to = re.search(r'(?:^|&&|;)\s*timeout\s+(\d+)', c)
        feats['cmd_has_timeout_prefix'] = int(m_to is not None)
        feats['cmd_timeout_n'] = int(m_to.group(1)) if m_to else 0
        # sleep N inline (server-style)
        m_sl = re.search(r'sleep\s+(\d+)', c)
        feats['cmd_has_sleep'] = int(m_sl is not None)
        feats['cmd_sleep_n'] = int(m_sl.group(1)) if m_sl else 0
        # hang hints
        feats['cmd_hang_hint'] = int(any(t in c.lower()
                                          for t in _HANG_HINT_TOKENS))

        # 子参数语义
        if 'pip install' in c:
            feats.update(_extract_pip_features(c))
        if 'pytest' in c:
            feats.update(_extract_pytest_features(c))
        if re.search(r'(?<!\\)\|', c):
            feats['cmd_pipe_count'] = len(re.findall(r'(?<!\\)\|(?!\|)', c))

    elif name == 'str_replace_editor':
        cmd = (args.get('command') or '').strip()
        feats['editor_cmd'] = cmd
        if cmd == 'create':
            feats['editor_filetext_len'] = len(args.get('file_text') or '')
        if cmd == 'str_replace':
            feats['editor_old_len'] = len(args.get('old_str') or '')
            feats['editor_new_len'] = len(args.get('new_str') or '')
        if cmd == 'view':
            feats['editor_has_range'] = int(
                isinstance(args.get('view_range'), list))

    return feats


# ---------------------------------------------------------------------------
# C: 同 trajectory 工件特征
# ---------------------------------------------------------------------------

class FileCache:
    """维护 trajectory 内已知文件内容 + testbed 结构。

    数据来源:
      - str_replace_editor::create  -> file_text (确定真值)
      - str_replace_editor::view    -> 从 OBSERVATION 解析 (可能截断)
      - str_replace_editor::str_replace -> 在已缓存内容上 apply
      - bash find -> 从 OBSERVATION 提取路径列表, 反映 testbed 结构
    """

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.testbed_paths: list[str] = []
        self.testbed_path_set: set[str] = set()

    def update(self, tool_call: dict, observation_text: str = '') -> None:
        fn = tool_call.get('function') or {}
        name = fn.get('name', '')
        args = _parse_args(fn.get('arguments'))

        if name == 'str_replace_editor':
            cmd = (args.get('command') or '').strip()
            path = args.get('path') or ''
            if cmd == 'create':
                self.files[path] = (args.get('file_text') or '')
            elif cmd == 'str_replace' and path in self.files:
                old = args.get('old_str') or ''
                new = args.get('new_str') or ''
                if isinstance(old, str) and isinstance(new, str) and old:
                    self.files[path] = self.files[path].replace(old, new, 1)
            elif cmd == 'view' and path and observation_text:
                content = self._parse_view_observation(observation_text)
                if content is not None and path not in self.files:
                    # view 只在 create 没缓存过时才存(create 是真值, view 是粗略)
                    self.files[path] = content
            elif cmd == 'insert' and path in self.files:
                # 简化: 插入操作不试图精确模拟, 标记已动过
                pass
            return

        if name == 'bash':
            cmd = (args.get('command') or '').strip()
            body = _strip_chmod_prefix(_strip_cd_prefix(cmd))
            if observation_text and re.search(r'^\s*find\b', body):
                # 解析 OBSERVATION, 提路径
                paths = []
                for line in observation_text.splitlines():
                    line = line.strip()
                    if line.startswith('/') and (
                            '.' in line or '/' in line[1:]):
                        paths.append(line)
                if paths:
                    # 累积式更新, 不覆盖
                    for p in paths:
                        if p not in self.testbed_path_set:
                            self.testbed_paths.append(p)
                            self.testbed_path_set.add(p)
            # 如果是 str_replace_editor view 通过 OBSERVATION 来的(含 cat -n)
            return

    @staticmethod
    def _parse_view_observation(text: str) -> Optional[str]:
        if not text or 'too large to display' in text.lower():
            return None
        # SWE-agent view OBSERVATION 格式:
        #   "Here's the result of running `cat -n` on /path:\n     1\tcontent\n..."
        out = []
        any_match = False
        for ln in text.splitlines():
            m = re.match(r'^\s*(\d+)\s*\t?\s*(.*)$', ln)
            if m:
                out.append(m.group(2))
                any_match = True
        return '\n'.join(out) if any_match else None


# C 类: 脚本内容信号. 经特征重要性诊断后只保留实际有用的几个.
# 移除的 (gain=0): sleep, while_true, input_call, bottle_run, fastapi_run,
# tornado_ioloop, http_server, socket_bind, itertools_count, asyncio_run
_HANG_SIGNALS = {
    'flask_run': re.compile(r'(?:^|\s|\.)\bapp\.run\s*\('),
    'serve_forever': re.compile(r'\.serve_forever\s*\('),
}
_NETWORK_PY_PATTERNS = (
    re.compile(r'\b(?:requests|urllib|urllib2|httpx|aiohttp|http\.client)\b'),
    re.compile(r'\bsocket\.'),
)


def extract_c_features(tool_call: dict, file_cache: FileCache) -> dict:
    """C 类: 同 trajectory 工件特征。仅对 bash 中"作用于已知文件"的命令产出。"""
    fn = tool_call.get('function') or {}
    if fn.get('name') != 'bash':
        return {}
    args = _parse_args(fn.get('arguments'))
    cmd = (args.get('command') or '').strip()
    body = _strip_chmod_prefix(_strip_cd_prefix(cmd))

    out: dict[str, Any] = {}

    # python 脚本 -> 看脚本内容
    m = re.search(r'python3?\s+(?:-\S+\s+)*([^\s|;&]+\.py)', body)
    if m:
        path = m.group(1)
        content = _resolve_file(file_cache, path)
        out['c_script_resolved'] = int(content is not None)
        if content is not None:
            lines = content.split('\n')
            out['c_script_loc'] = len(lines)
            out['c_script_n_imports'] = sum(
                1 for ln in lines
                if ln.lstrip().startswith(('import ', 'from ')))
            # 每种 hang 模式独立信号 (让模型区分严重程度)
            n_hang = 0
            for sig_name, pat in _HANG_SIGNALS.items():
                hit = int(bool(pat.search(content)))
                out[f'c_script_{sig_name}'] = hit
                n_hang += hit
            out['c_script_n_hang_signals'] = n_hang
            # "强 hang" 标记: 有 server-style 模式之一
            out['c_script_strong_hang'] = int(any(
                pat.search(content)
                for name, pat in _HANG_SIGNALS.items()
                if name in ('flask_run', 'fastapi_run', 'tornado_ioloop',
                            'serve_forever', 'http_server', 'bottle_run')))
            out['c_script_has_network'] = int(
                any(p.search(content) for p in _NETWORK_PY_PATTERNS))
            out['c_script_has_subprocess'] = int(
                'subprocess' in content)
            out['c_script_has_threading'] = int(
                'threading' in content or 'multiprocessing' in content)

    # pip install -e -> 看 setup.py / pyproject.toml
    if 'pip install' in body and re.search(r'(^|\s)-e\b', body):
        for path in ('/testbed/setup.py', '/testbed/pyproject.toml',
                     'setup.py', 'pyproject.toml'):
            content = _resolve_file(file_cache, path)
            if content is None:
                continue
            out['c_pip_setup_resolved'] = 1
            out['c_pip_setup_loc'] = len(content.split('\n'))
            if path.endswith('setup.py'):
                m_dep = re.search(r'install_requires\s*=\s*\[(.*?)\]',
                                  content, re.DOTALL)
                out['c_pip_n_deps'] = (
                    m_dep.group(1).count(',') + 1 if m_dep else 0)
                out['c_pip_has_cython'] = int(
                    'Cython' in content or 'cython' in content
                    or '.pyx' in content)
                out['c_pip_has_c_ext'] = int(bool(
                    re.search(r'Extension\s*\(', content)))
            elif path.endswith('pyproject.toml'):
                # toml 简单提依赖数
                m_dep = re.search(r'dependencies\s*=\s*\[(.*?)\]', content,
                                  re.DOTALL)
                out['c_pip_n_deps'] = (
                    m_dep.group(1).count(',') + 1 if m_dep else 0)
            break

    # pytest -> testbed 结构
    if 'pytest' in body and file_cache.testbed_paths:
        n_test_files = sum(1 for p in file_cache.testbed_paths
                           if p.endswith('.py')
                           and ('test' in p.lower()))
        out['c_pytest_n_test_files'] = n_test_files
        out['c_pytest_total_files'] = len(file_cache.testbed_paths)

    return out


def _resolve_file(cache: FileCache, path: str) -> Optional[str]:
    """灵活查 cache: 直接命中 / /testbed 前缀 / 后缀匹配。"""
    if path in cache.files:
        return cache.files[path]
    # 加 /testbed 前缀试一次
    if not path.startswith('/'):
        for prefix in ('/testbed/', '/'):
            cand = prefix + path
            if cand in cache.files:
                return cache.files[cand]
    # 后缀匹配 (如 'reproduce.py' 匹配 '/testbed/reproduce.py')
    for cached_path, content in cache.files.items():
        if cached_path.endswith('/' + path) or cached_path.endswith(path):
            return content
    return None


# ---------------------------------------------------------------------------
# E: 轨迹历史观测特征
# ---------------------------------------------------------------------------

class TrajectoryState:
    """维护一条 trajectory 内已观测到的工具调用真实时长。

    用法 (训练或推理都一样):
        state = TrajectoryState(global_bucket_log_means)
        for tool_call in trajectory:
            features = state.extract(tool_call)
            ... 训练: 写一行 (features, actual_t); 推理: 调模型预测 ...
            state.observe(tool_call, actual_t)
    """

    def __init__(self,
                 global_bucket_log_means: dict[str, float],
                 timeout_threshold_s: float = 60.0):
        # global_bucket_log_means[bucket] = log(1 + bucket_train_median)
        self._mu = global_bucket_log_means
        self._timeout_thresh = timeout_threshold_s
        # 历史状态 (永远只反映已 observe() 的)
        self._bucket_history: dict[str, list[float]] = defaultdict(list)
        self._all_t: list[float] = []
        self._heavy_t: list[tuple[str, float]] = []   # (bucket, t)
        self._n_observed = 0
        self._has_seen_timeout = False
        self._cum_bash_time = 0.0
        # C 类: 文件 / testbed 结构缓存
        self._file_cache = FileCache()
        # E5 类: 从最近一次 OBSERVATION 提取的信号
        self._last_obs_signals: dict[str, int] = {}
        self._cum_obs_signals: dict[str, int] = defaultdict(int)

    @staticmethod
    def _is_heavy_bucket(b: str) -> bool:
        # 重命令: pip install / python script / pytest / unittest / mypy / setup
        # 这类观测对推断 trajectory 速度因子有强信号
        if b.startswith('bash::pip::'):
            return True
        if b.startswith('bash::python::'):
            return ('script' in b or 'pytest' in b or 'unittest' in b
                    or 'mypy' in b or 'setup' in b or 'module' in b)
        if b.startswith('bash::conda::') or b.startswith('bash::apt::'):
            return True
        if b.startswith('bash::env_prefix') or b.startswith('bash::bg_server'):
            return True
        return False

    def extract(self, tool_call: dict) -> dict:
        feats = _extract_static_features(tool_call)
        b = feats['bucket']

        # E1: 同桶历史
        hist = self._bucket_history.get(b, [])
        feats['e1_same_bucket_count'] = len(hist)
        if hist:
            log_h = [math.log1p(t) for t in hist]
            feats['e1_same_bucket_log_mean'] = sum(log_h) / len(log_h)
            feats['e1_same_bucket_log_std'] = (
                _std(log_h) if len(log_h) >= 2 else 0.0)
            feats['e1_same_bucket_last'] = math.log1p(hist[-1])
        else:
            feats['e1_same_bucket_log_mean'] = float('nan')
            feats['e1_same_bucket_log_std'] = float('nan')
            feats['e1_same_bucket_last'] = float('nan')

        # E2: 异桶迁移 — trajectory 速度因子
        # r_i = log(1+t_i) - log(1+μ_b(i)), 平均
        ratios = []
        for bk, t in self._heavy_t:
            mu = self._mu.get(bk)
            if mu is None or mu <= 0:
                continue
            ratios.append(math.log1p(t) - mu)
        if ratios:
            feats['e2_traj_log_speed_factor'] = sum(ratios) / len(ratios)
            feats['e2_n_heavy_observed'] = len(ratios)
        else:
            feats['e2_traj_log_speed_factor'] = float('nan')
            feats['e2_n_heavy_observed'] = 0

        # E3: 全部观测 (含轻命令) 的 log 均值, 作为机器/磁盘速度代理
        if self._all_t:
            feats['e3_all_log_mean'] = (
                sum(math.log1p(t) for t in self._all_t) / len(self._all_t))
            feats['e3_n_all_observed'] = len(self._all_t)
        else:
            feats['e3_all_log_mean'] = float('nan')
            feats['e3_n_all_observed'] = 0

        # E4: 全局 trajectory 状态
        feats['e4_round_k'] = self._n_observed
        feats['e4_cum_bash_time'] = self._cum_bash_time
        feats['e4_seen_timeout'] = int(self._has_seen_timeout)

        # E5: 上一次 OBSERVATION 的状态信号 (只留特征重要性诊断后有用的几个).
        # 移除: killed, timeout_error, oom (训练数据里没出现这些信号 -> gain=0)
        for sig in ('traceback', 'server_running', 'building'):
            feats[f'e5_last_obs_{sig}'] = int(
                self._last_obs_signals.get(sig, 0))
            feats[f'e5_cum_obs_{sig}'] = self._cum_obs_signals.get(sig, 0)

        # C 类: 工件特征 (基于已观测到的 file_cache)
        feats.update(extract_c_features(tool_call, self._file_cache))

        return feats

    def observe(self, tool_call: dict, actual_t: float,
                observation_text: str = '') -> None:
        """把 (tool_call, actual_t) 写入历史。必须在 extract() 之后调。

        observation_text 是这次工具调用对应的 OBSERVATION 文本(若有),
        用于更新 C 类 file_cache。
        """
        b = classify(tool_call)
        self._bucket_history[b].append(actual_t)
        self._all_t.append(actual_t)
        if self._is_heavy_bucket(b):
            self._heavy_t.append((b, actual_t))
        self._n_observed += 1
        if (tool_call.get('function') or {}).get('name') == 'bash':
            self._cum_bash_time += actual_t
        if actual_t >= self._timeout_thresh:
            self._has_seen_timeout = True
        # 更新 file cache
        self._file_cache.update(tool_call, observation_text)
        # E5: 解析 OBSERVATION 状态信号
        self._update_obs_signals(observation_text)

    def _update_obs_signals(self, obs: str) -> None:
        if not obs:
            self._last_obs_signals = {}
            return
        sigs = {
            'killed': bool(re.search(
                r'\b(?:Killed|killed by signal|got signal\s+\d+)\b', obs)),
            'traceback': 'Traceback (most recent call last)' in obs,
            'timeout_error': bool(re.search(
                r'\b(?:TimeoutError|Timed out|timed out)\b', obs,
                re.IGNORECASE)),
            'server_running': bool(re.search(
                r'(?:Listening on|Running on http|Started server|'
                r'\* Serving Flask app|Uvicorn running|'
                r'(?:0\.0\.0\.0|127\.0\.0\.1|localhost):\d+)', obs)),
            'building': bool(re.search(
                r'(?:Building wheel|Building source|Compiling|'
                r'Compiling extensions|Running setup\.py)', obs)),
            'oom': bool(re.search(
                r'\b(?:OOMKilled|out of memory|MemoryError|Cannot allocate)\b',
                obs, re.IGNORECASE)),
        }
        self._last_obs_signals = {k: int(v) for k, v in sigs.items()}
        for k, v in sigs.items():
            if v:
                self._cum_obs_signals[k] += 1


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)
