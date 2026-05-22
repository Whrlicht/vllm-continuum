"""工具调用桶分类器 (v3, 纯规则)。

输入: 一个 tool_call dict (含 function.name 和 function.arguments JSON 字符串)。
输出: 一个字符串桶 ID, 例如 'bash::python::script_repro'。

桶 ID 是预测器的核心 categorical 特征。规则覆盖 SWE-agent 风格 trace, 但
设计上跨数据集——不依赖任何 instance_id / repo / model 名。
"""

from __future__ import annotations

import json
import re
from typing import Any

ENV_VAR_PREFIX = re.compile(r'^[A-Z_][A-Z_0-9]*=')


def _parse_args(args_raw: Any) -> dict:
    if isinstance(args_raw, dict):
        return args_raw
    if isinstance(args_raw, str):
        try:
            return json.loads(args_raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _strip_cd_prefix(c: str) -> str:
    """剥掉 'cd /path && ...' 前缀, 返回真实命令体。"""
    if not c:
        return c
    first = c.split()[0]
    if first == 'cd' and '&&' in c:
        return c.split('&&', 1)[1].strip()
    return c


def _strip_chmod_prefix(body: str) -> str:
    """剥掉 'chmod +x /file && ...', chmod 自身耗时小, 后面才是真命令。"""
    bf = body.split()[0] if body else ''
    if bf == 'chmod' and '&&' in body:
        return body.split('&&', 1)[1].strip()
    return body


_LIGHT_CMDS = {
    'grep', 'rm', 'ls', 'cat', 'head', 'tail', 'wc', 'sed', 'awk', 'mkdir',
    'sleep', 'kill', 'pkill', 'source', 'curl', 'wget', 'make', 'patch',
    'unzip', 'cp', 'mv', 'ps', 'echo', 'man', 'chmod',
}


def _classify_bash(c: str) -> str:
    if not c.strip():
        return 'bash::empty'

    has_amp = '&&' in c
    has_pipe = bool(re.search(r'(?<!\\)\|(?!\|)', c))
    has_bg = bool(re.search(r'(?<!&)&(?:\s|$)', c))

    if has_bg and re.search(r'sleep\s+\d+', c):
        return 'bash::bg_server'

    body = _strip_chmod_prefix(_strip_cd_prefix(c))
    bf = body.split()[0] if body else ''

    # pip 子分类
    if 'pip install' in body:
        m = re.search(r'pip\s+install\s+([^&|;]+)', body)
        target = m.group(1).strip() if m else ''
        is_e = bool(re.search(r'(^|\s)-e\b', target))
        has_x = bool(re.search(r'\[[a-zA-Z,_-]+\]', target))
        target_no_e = target.replace('-e', '').strip()
        is_local = bool(re.search(r'(^|\s)(\.|/[^\s]+|[^/\s]+/)(\s|$)',
                                  target_no_e))
        no_build_iso = '--no-build-isolation' in target
        no_deps = '--no-deps' in target
        has_venv = bool(re.search(r'(python\s*-m\s*venv|virtualenv)', c))
        has_uninstall = 'pip uninstall' in c

        if is_e or is_local:
            family = ('editable_extras' if (is_e and has_x) else
                      'editable_local' if is_e else 'local_path')
            if no_build_iso and no_deps:
                sub = 'no_build_iso_no_deps'
            elif no_build_iso:
                sub = 'no_build_iso'
            elif no_deps:
                sub = 'no_deps'
            else:
                sub = 'default'
            return f'bash::pip::{family}::{sub}'

        if has_venv:
            return 'bash::pip::pypi::with_venv_create'
        if has_uninstall:
            return 'bash::pip::pypi::with_uninstall'
        if has_x:
            return 'bash::pip::pypi::single_with_extras'
        toks = re.split(r'\s+', target)
        n_pkgs = sum(1 for tk in toks if tk and not tk.startswith('-'))
        return ('bash::pip::pypi::multi_pkg' if n_pkgs >= 2
                else 'bash::pip::pypi::single')

    if 'pip uninstall' in body:
        return 'bash::pip::uninstall'
    if bf == 'pip':
        return 'bash::pip::other'

    # python 子分类
    if re.match(r'^python3?\b', bf):
        if 'mypy' in body:
            return 'bash::python::module_mypy'
        if 'pytest' in body:
            if re.search(r'-k\s', body):
                return 'bash::python::pytest_keyword'
            ptest = re.search(r'pytest\b\s*(.*)', body)
            args_after = ptest.group(1).strip() if ptest else ''
            args_clean = re.sub(r'-\S+', '', args_after).strip()
            if not args_clean:
                return 'bash::python::pytest_full_discovery'
            if '::' in args_clean:
                return 'bash::python::pytest_single_test'
            return 'bash::python::pytest_path'
        if 'unittest' in body:
            disc = bool(re.search(r'unittest\s+discover\b', body))
            return ('bash::python::unittest_discover' if disc
                    else 'bash::python::unittest_specific')
        if re.search(r'setup\.py\s+install', body):
            return 'bash::python::setup_install'
        if re.search(r'setup\.py\s+(?:develop|build)', body):
            return 'bash::python::setup_dev_build'
        if re.search(r'-m\s+(?:line_profiler|kernprof)\b', body):
            return 'bash::python::module_profiler'
        if ' -m ' in body:
            return 'bash::python::module_other'
        if re.search(r'^python3?\s+-c\b', body):
            ln = len(body)
            if ln < 150:
                return 'bash::python::interactive_short'
            if ln < 400:
                return 'bash::python::interactive_medium'
            return 'bash::python::interactive_long'
        if '.py' in body:
            if has_bg:
                return 'bash::python::script_bg'
            has_setup = ('pip install' in c or 'apt-get' in c
                         or 'conda install' in c
                         or re.search(r'python\s*-m\s*venv', c))
            if has_setup:
                return 'bash::python::script_with_setup'
            if re.search(r'(repro|reproduce|test_)', body):
                return 'bash::python::script_repro'
            return 'bash::python::script_other'
        return 'bash::python::interactive_misc'

    # 直接 pytest (不通过 python -m)
    if bf == 'pytest':
        return 'bash::pytest_direct'

    # find 子分类
    if bf == 'find':
        has_exec_grep = bool(re.search(r'-exec\s+grep', body))
        is_dir_only = '-type d' in body
        has_name = '-name' in body
        if has_exec_grep:
            return 'bash::find::exec_grep'
        if is_dir_only:
            return 'bash::find::dir_only'
        if has_name and has_pipe:
            return 'bash::find::with_name_pipe'
        if has_name:
            return 'bash::find::with_name_basic'
        return 'bash::find::other'

    # conda / apt / git
    if bf == 'conda':
        toks = body.split()
        return f'bash::conda::{toks[1] if len(toks) > 1 else "unknown"}'
    if bf in ('apt', 'apt-get'):
        toks = body.split()
        return f'bash::apt::{toks[1] if len(toks) > 1 else "unknown"}'
    if bf == 'git':
        toks = body.split()
        return f'bash::git::{toks[1] if len(toks) > 1 else "unknown"}'

    # env 变量前缀 (PYTHONPATH=... cmd ...)
    if ENV_VAR_PREFIX.match(bf):
        return 'bash::env_prefix'

    # 轻命令
    if bf in _LIGHT_CMDS:
        chain_tag = '::chained' if has_amp else ''
        return f'bash::light::{bf}{chain_tag}'

    if bf == 'cd':
        return 'bash::cd_only'

    return f'bash::other_first::{bf[:24]}'


def classify(tool_call: dict) -> str:
    """主入口: 返回桶 ID。"""
    fn = tool_call.get('function') or {}
    name = fn.get('name', '')
    if name == 'submit':
        return 'submit'
    if name == 'str_replace_editor':
        args = _parse_args(fn.get('arguments'))
        cmd = (args.get('command') or '').strip()
        if cmd == 'view':
            return ('editor::view::with_range'
                    if isinstance(args.get('view_range'), list)
                    else 'editor::view::no_range')
        if cmd == 'create':
            sz = len(args.get('file_text') or '')
            bk = ('lt1KB' if sz < 1024
                  else 'lt10KB' if sz < 10240
                  else 'lt50KB' if sz < 51200 else 'ge50KB')
            return f'editor::create_{bk}'
        if cmd == 'str_replace':
            tot = len(args.get('old_str') or '') + len(args.get('new_str') or '')
            bk = ('lt200' if tot < 200
                  else 'lt1000' if tot < 1000
                  else 'lt5000' if tot < 5000 else 'ge5000')
            return f'editor::str_replace_{bk}'
        if cmd == 'insert':
            return 'editor::insert'
        return f'editor::{cmd or "unknown"}'
    if name == 'bash':
        args = _parse_args(fn.get('arguments'))
        return _classify_bash((args.get('command') or '').strip())
    return f'unknown::{name}'


# 桶大类 (用于决定是走查表还是走 ML)
def family(bucket: str) -> str:
    if bucket == 'submit':
        return 'submit'
    if bucket.startswith('editor::'):
        return 'editor'
    if bucket.startswith('bash::'):
        return 'bash'
    return 'unknown'
