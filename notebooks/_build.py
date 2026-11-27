"""Build the workshop notebooks with nbformat: python notebooks/_build.py"""

from __future__ import annotations

import pathlib

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).parent
NOTEBOOK = HERE / "workshop_01_agents_that_train_themselves.ipynb"


def md(text: str):
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


CELLS = [
    md("""
# Agents that train themselves

In this notebook an AI agent runs a reinforcement learning (RL) project by itself. In RL, a
program called a policy learns to control something by trial and error: it tries actions, gets a
score for how well they went, and slowly learns which actions earn the best scores. Usually a
person sets up that loop. They decide how to score the policy, train it, check whether it
actually does the job, and adjust things when it doesn't.

Here an agent does all of that. It gets one goal, "land the lunar lander gently on the pad". From
there it writes the scoring rule (the reward function), trains a policy with it, has the policy
graded by an evaluator it can't see, works out from the grades what went wrong, and tries again
until the lander lands.

![The untrained lander firing its engines at random, drifting and tilting until it crashes](images/random.gif)

This is the lander before any training, firing its engines at random. It drifts, tilts and
crashes, because nothing has told it what a good flight looks like yet. The pad is the flat
stretch between the two flags. By the end of the notebook, the agent will have trained a policy
that brings the lander down gently between them.

## 1. Before you start

A Jupyter notebook is a page of cells. Text cells like this one explain what's going on, and
code cells run Python. To run a code cell, click the play button next to it, or click into it
and press Shift+Enter. While it runs you'll see `[*]` next to it, and its output appears
underneath when it's done. Run the cells from top to bottom.

- The kernel is the Python process that runs the cells. If you're asked to pick one, choose
  the `.venv` environment.
- A line that starts with `!` is a shell command, the same as typing it in a terminal.
- A cell that starts with `%%writefile <file>` saves the cell to that file instead of running
  it, and `%%writefile -a <file>` adds the cell to the end of the file. `%run -n <file>` loads a
  Python file into the notebook without running its main part. Section 3 uses all three.
- If a cell hangs or things get confused, restart the kernel and run the cells again from the
  top.
- If you set `LIVE = False` in the first code cell (section 3), nothing calls Amazon Bedrock or
  starts a box, and the notebook uses saved results from real runs in `results/`.

## 2. What you'll use

**Strands harness.** The agent is the [Strands harness](https://strandsagents.com/docs/user-guide/harness/)
default agent. It's a Strands `Agent` that already has shell, file and web tools, memory,
sessions and background tasks, so we only add the task, one tool and a stop rule. It runs on
Amazon Bedrock with the harness's default models.

**Strands Box.** [Strands Box](https://strandsagents.com/docs/user-guide/box/) is a sandbox for agents.
The agent runs inside it, and Box checks every file it opens, program it starts and network
request it makes against a policy. We use it to keep the success criteria out of the agent's
reach and to cap how often it can be graded.

**Dogwood.** That policy is written in [Dogwood](https://github.com/dogwood-policy/dogwood), a
policy language built on Cedar that adds rules over time, such as "refuse this request once
six like it have succeeded". Anything no rule permits is refused.

None of this needs special hardware. Training is a small network on the CPU, so the workshop
runs on a laptop as long as you have access to Bedrock.

### 2.1 RL in two minutes

The environment is [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
from [Gymnasium](https://gymnasium.farama.org/), a small 2D physics simulation. At each step the
policy gets an observation of 8 numbers: the lander's position relative to the pad (`x`, `y`),
its velocity, angle and angular velocity, and whether each leg touches the ground. It picks one
of four actions: do nothing, fire the left engine, the main engine, or the right engine. An
episode is one flight, ending in a landing, a crash, or a timeout after 1,000 steps.

After each step the environment hands back a reward, a number, and training tries to maximize
the total over an episode. The policy here is a small neural network (two layers of 64 units).
[Proximal Policy Optimization](https://en.wikipedia.org/wiki/Proximal_policy_optimization) (PPO) from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
trains it by flying thousands of episodes and nudging the network towards actions that earned
more reward.

The same approach is used to train robots to walk and grasp, to play games such as Go, to
control data centre cooling, to race cars in AWS DeepRacer, and to fine-tune large language
models from human feedback.

Training only ever sees the reward. If the reward pays for the wrong thing, training learns the
wrong thing. Here the agent writes the reward, and we check its policies with a separate
evaluation.

## 3. Build the agent

The agent runs inside a box as its own program, `agent.py`. The cells in this section write that
program one piece at a time, so you can read each piece before it runs.

Run the next cell first. It does some setup: it moves into the repo's top folder, so file paths
like `results/` work, makes sure the notebook can find the right `python` and the `box` command,
and starts the grader (more on that in section 6). Then it makes a new run folder for this run,
for example `runs/20261013-101500`, and saves its name as `RUN`. Everything for your agent goes
in there.

The settings are in the middle of the cell: `LIVE` and the Bedrock region. You don't
need to change them for the workshop.
"""),
    code("""
import os
import pathlib
import subprocess
import sys
import time

from rl.grader_client import ensure_grader

if pathlib.Path.cwd().name == "notebooks":
    os.chdir("..")
os.environ["PATH"] = os.pathsep.join([sys.prefix + "/bin", str(pathlib.Path.home() / ".local/bin"), os.environ["PATH"]])
REPO = pathlib.Path.cwd()

# Set LIVE = False to use the saved results in results/ without calling Bedrock or starting a box.
LIVE = True
REGION = "us-west-2"

ensure_grader()
RUN = time.strftime("runs/%Y%m%d-%H%M%S")
os.makedirs(RUN)
print("run folder:", RUN)
"""),
    md("""
`agent.py` starts with the imports the whole program uses:
"""),
    code(r'''
%%writefile {RUN}/agent.py
import json
import os
import pathlib
import sys

from strands import tool
from strands.hooks import AfterToolCallEvent, AfterToolsEvent, HookProvider, HookRegistry
from strands_harness import create_harness

import rl
from rl.grader_client import evaluate
'''),
    md("""
### 3.1 The evaluation tool

The agent gets one [tool](https://strandsagents.com/docs/user-guide/sdk/tools/) of its own.
`evaluate_policy` sends a trained policy's weights to the grader and returns the metrics, and it
logs every result to `artifacts/eval_log.jsonl` in the agent's folder. It's the agent's only way
to the success criteria.

The `@tool` decorator turns a plain Python function into a Strands tool. Strands describes the
tool to the model (the LLM) with three things taken from the function: its name
(`evaluate_policy`), its description (the docstring's first paragraph), and an input schema
built from the parameters' type hints, defaults and the `Args` section of the docstring. The
model reads those to decide when to call the tool and what to pass, so the docstring is written
for the model.
"""),
    code(r'''
%%writefile -a {RUN}/agent.py


@tool
def evaluate_policy(run_dir: str, n_episodes: int = 50) -> dict:
    """Evaluate a trained LunarLander policy against the sealed success criteria, on held-out episodes.

    Args:
        run_dir: Run directory holding the trained policy, e.g. "artifacts/iter_3".
        n_episodes: Number of held-out episodes to evaluate (default 50, at most 200).

    Returns:
        dict with landing_rate, crash_rate, timeout_rate, loiter_fraction, mean_episode_steps,
        and passed (bool).
    """
    result = evaluate(run_dir, n_episodes)
    with open("artifacts/eval_log.jsonl", "a") as log:
        log.write(json.dumps({"run_dir": run_dir, **result}) + "\n")
    return result
'''),
    md("""
### 3.2 The stop rule

When to stop is a deterministic check, so it's a Strands [hook](https://strandsagents.com/docs/user-guide/sdk/agents/hooks/)
instead of something the model decides. Hooks are callbacks the agent loop runs at fixed points.
`StopRule` reads the result of every `evaluate_policy` call
([`AfterToolCallEvent`](https://strandsagents.com/docs/api/python/strands.hooks.events/#aftertoolcallevent)),
and after each batch of tool calls
([`AfterToolsEvent`](https://strandsagents.com/docs/api/python/strands.hooks.events/#aftertoolsevent))
it ends the run if the latest result meets the target, the budget is used up, or the last three
results are within 0.05 of each other. Setting
`end_turn` stops the loop before the next model call, and the model can't skip or misreport it.
"""),
    code(r'''
%%writefile -a {RUN}/agent.py


class StopRule(HookProvider):
    """Ends the agent's turn on the target, an exhausted budget, or a plateau."""

    def __init__(self, target: float, budget: int):
        self.target = target
        self.budget = budget
        self.rates: list[float] = []

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolCallEvent, self.record)
        registry.add_callback(AfterToolsEvent, self.check)

    def record(self, event: AfterToolCallEvent) -> None:
        if event.tool_use["name"] != "evaluate_policy" or event.result.get("status") != "success":
            return
        for block in event.result.get("content", []):
            try:
                metrics = json.loads(block.get("text", ""))
            except ValueError:
                continue
            if isinstance(metrics, dict) and isinstance(metrics.get("landing_rate"), int | float):
                self.rates.append(float(metrics["landing_rate"]))

    def check(self, event: AfterToolsEvent) -> None:
        if not self.rates:
            return
        latest, best, window = self.rates[-1], max(self.rates), self.rates[-3:]
        if latest >= self.target:
            reason = f"target met: {latest:.2f} >= {self.target:.2f}"
        elif len(self.rates) >= self.budget:
            reason = f"evaluation budget used ({len(self.rates)}/{self.budget}); best {best:.2f}"
        elif len(window) == 3 and max(window) - min(window) < 0.05:
            reason = f"plateau: last 3 landing rates within 0.05; best {best:.2f}"
        else:
            return
        print(f"\nstop rule: {reason}", flush=True)
        event.end_turn = f"Stopped by the stop rule: {reason}"
'''),
    md("""
### 3.3 The instructions

The instructions describe a method that would work for any domain: propose a candidate, run it,
evaluate it against the sealed criteria, diagnose, and go again. Sealed means the agent can't see
or change the criteria; sections 6 and 7 explain how that works.

`create_harness` appends these instructions to the Strands harness's own
[system prompt](https://github.com/strands-agents/harness-sdk/blob/main/harness-py/src/strands_harness/prompt.py),
which covers how the harness agent works in general.
"""),
    code(r'''
%%writefile -a {RUN}/agent.py


INSTRUCTIONS = """You are an autonomous optimization researcher. You get an objective and a set of tools, and you
run the whole experiment loop yourself with no human involved. Propose a candidate, run the
experiment, evaluate the result against sealed quantitative criteria, work out from the results
why it fell short, and try again with a changed candidate or changed hyperparameters. Stop when
you meet the target or a stop condition applies.

# Method

1. Propose. Turn the objective into a concrete candidate you can test. For a reinforcement
   learning task the candidate is a reward function; for other tasks it might be a
   configuration or a program. Follow the authoring rules in the task brief.
2. Run. Start the experiment, for example a training run, and wait for it to finish.
3. Evaluate. Score the candidate with the evaluation tool. Its results are the ground truth.
4. Diagnose. Use the returned metrics to explain why the candidate falls short.
5. Iterate. Write an improved candidate and go back to step 1. The run ends by itself when a
   stop condition holds.

# Principles

- You cannot read or run the success criteria. The evaluation tool is the only way to learn
  how a candidate scores, so don't try to inspect, import or reverse-engineer it. A better score
  should come from a better candidate.
- Your candidate's own signal, such as training reward, can go up while the sealed criteria
  stay unmet. That usually means the candidate is gaming its proxy. When the two disagree,
  believe the sealed evaluation and fix the candidate.
- The run stops when you hit the target, when the iteration budget runs out, or when three
  results in a row are within 0.05 of each other. Make each evaluation count.
- Keep each iteration's files separate so the run can be audited afterwards.
- Keep your commentary short and report the metrics.

The method doesn't depend on the domain. It works the same for robotics, games or control
systems; only the task brief changes (the environment, the authoring rules and the target)."""
'''),
    md("""
### 3.4 The objective

The task itself is in the objective: the prompt, a single request, that the harness agent gets
when it's invoked. It describes the environment, the rules for writing a reward, how to train,
and the target and budget. To point the same agent at a different problem, you change this text
and the grader.
"""),
    code(r'''
%%writefile -a {RUN}/agent.py


OBJECTIVE = """Objective: train a policy that lands the lander gently on the pad.

Task: Gymnasium LunarLander-v3 (discrete). Your current directory is your workspace; keep all your
files in it. The training code is the `rl` package; you may read {rl_dir}/env.py and
{rl_dir}/guard.py to see how your reward is wrapped and checked.
- Observation is an 8-vector [x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact];
  x, y are position relative to the pad (pad is approximately the origin), legs are 1.0 on contact
  else 0.0.
- Actions: 0 = do nothing, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine.

Reward-authoring contract: write each candidate reward to rewards/iter_<N>.py as a function named
exactly `reward(obs, action, next_obs, terminated, truncated, info)` returning a float. Only
arithmetic, comparisons, and calls to these functions are allowed:
  abs min max sum len float int pow round bool
  math.sqrt math.exp math.log math.tanh math.sin math.cos math.atan2 math.hypot math.fabs math.isfinite
  np.abs np.sqrt np.clip np.tanh np.exp np.log np.square np.sign np.minimum np.maximum np.hypot
  np.isfinite np.sum np.mean np.array np.asarray np.linalg.norm
The constants math.pi, math.e, math.inf, np.pi and np.inf are allowed. No imports, no helper
functions, no other calls or attributes of math or np, no methods on values, and no attributes
beginning with an underscore. Training rejects violations.

Run a candidate (hyperparameters are tunable; adjust them alongside the reward):
  {python} -m rl.train --reward rewards/iter_<N>.py --steps {steps} --seed {seed} --out artifacts/iter_<N> \\
    [--gamma 0.999] [--ent-coef 0.01] [--lr 0.0003] [--n-steps 1024]
Use exactly that Python executable. `--steps` can be at most {steps}; training refuses more.
Defaults are shown in brackets. Then evaluate with evaluate_policy(run_dir="artifacts/iter_<N>").

Sealed metrics returned: landing_rate, crash_rate, timeout_rate, loiter_fraction,
mean_episode_steps, passed.

Target landing rate: {target:.2f}. Iteration budget: {budget} evaluations; the sandbox refuses any
beyond that. The run stops by itself after an evaluation that meets the target, uses the last of the
budget, or makes three results in a row within 0.05 of each other."""


def objective(seed: int, budget: int, target: float, steps: int) -> str:
    rl_dir = pathlib.Path(rl.__file__).resolve().parent
    return OBJECTIVE.format(rl_dir=rl_dir, python=sys.executable, steps=steps, seed=seed, target=target, budget=budget)
'''),
    md("""
### 3.5 The harness

This is the agent itself: one
[`create_harness`](https://strandsagents.com/docs/user-guide/harness/reference/configuration/)
call with the instructions, the tool and the hook. Everything else, the models included, is the
harness's defaults. At the bottom, `agent.py` reads its
settings (seed, budget, target, step ceiling) from environment variables, which the box
sets, moves into its workspace and gives the agent the objective.
"""),
    code(r'''
%%writefile -a {RUN}/agent.py


def build_agent(target: float, budget: int):
    return create_harness(
        instructions=INSTRUCTIONS,
        tools=[evaluate_policy],
        hooks=[StopRule(target, budget)],
    )


if __name__ == "__main__":
    budget = int(os.environ["AUTORL_BUDGET"])
    target = float(os.environ["AUTORL_TARGET"])
    os.chdir(os.environ["AUTORL_WORKSPACE"])
    with build_agent(target, budget) as agent:
        agent(objective(int(os.environ["AUTORL_SEED"]), budget, target, int(os.environ["AUTORL_MAX_STEPS"])))
'''),
    md("""
`agent.py` is complete. Loading it here lets us build the agent and look at it without starting
it. These are its tools, the harness's defaults plus ours:
"""),
    code("""
%run -n {RUN}/agent.py
agent = build_agent(target=0.90, budget=6)
print(agent.tool_names)
"""),
    md("And this is the objective it will start with:"),
    code("print(objective(seed=1, budget=6, target=0.90, steps=300_000))"),
    md("The whole of `agent.py`, as the box will run it:"),
    code("""
from IPython.display import Code

Code(filename=f"{RUN}/agent.py")
"""),
    md("""
## 4. Put it in a box

The agent can run shell commands, write and run code, and reach the web, so it runs inside a
sandbox. Strands Box starts the agent's process with only the files, programs and network access
we grant, and checks everything the agent does through its shell and tools against a policy.
Anything the policy doesn't allow is refused.

Box takes two files per agent. `box.toml` says what the agent's own process can reach: which
files, which programs and which credentials. `policy.dw` is the Dogwood policy that decides every
shell command, file operation and network request made through the box. Here they're templates,
with `$NAME` placeholders that get filled in for each agent. Section 7 shows the full files for
your agent and explains what each part does.
"""),
    code(r'''
from string import Template

BOX_TOML = Template("""
name = "$NAME"
box_dir = "$BOX_DIR"
policy = "policy.dw"

# Training runs take minutes, and Box stops a shell call after 30 seconds by default.
[limits]
shell.timeout = 1800

# The agent: agent.py, run with the venv's Python.
[agent]
command = ["$PYTHON", "-P", "$SCRIPT"]
workspace = "$WORKSPACE"

[agent.env]
AWS_REGION = "$REGION"
AWS_DEFAULT_REGION = "$REGION"
HOME = "$WORKSPACE"
TMPDIR = "$WORKSPACE/.tmp"
PATH = "/usr/bin:/bin"
GRADER_URL = "$GRADER_URL"
AUTORL_GRADER_VIA_PROXY = "1"
# The agent's output goes to a log file, which Python would otherwise write in 8 KB blocks.
PYTHONUNBUFFERED = "1"
AUTORL_WORKSPACE = "$WORKSPACE"
AUTORL_SEED = "$SEED"
AUTORL_BUDGET = "$BUDGET"
AUTORL_TARGET = "$TARGET"
AUTORL_MAX_STEPS = "$STEPS"

# Files the agent's own process can reach. The criteria (rl/evaluate.py and its compiled copy)
# and the example solution are denied, and nothing else in the repo is granted.
[agent.filesystem]
read = ["$WORKSPACE", "$SCRIPT", "$REPO/rl", "$VENV", "$PYTHON_BASE"$LINUX_READ]
exec = [$LINUX_EXEC]
write = ["$WORKSPACE"]
deny = ["$REPO/rl/evaluate.py", "$REPO/rl/__pycache__", "$REPO/rl/rewards/examples/landing.py"]

# Real CPython for training and the agent's own scripts. A bare `python` in the box's shell is
# Monty, a restricted interpreter, so the agent runs this one by its full path.
[tool.python]
command = ["$PYTHON"]

[tool.python.env]
HOME = "$WORKSPACE"
TMPDIR = "$WORKSPACE/.tmp"
AUTORL_MAX_STEPS = "$STEPS"

[tool.python.filesystem]
read = ["$WORKSPACE", "$REPO/rl", "$VENV", "$PYTHON_BASE"]
write = ["$WORKSPACE"]
deny = ["$REPO/rl/evaluate.py", "$REPO/rl/__pycache__", "$REPO/rl/rewards/examples/landing.py"]

# The agent gets a placeholder key; the box swaps in the real one on requests to Bedrock.
[egress.model]
destinations = ["$MODEL_HOST"]
secret.ref = "env://AWS_BEARER_TOKEN_BEDROCK"

# The harness's web_search tool calls Amazon Bedrock AgentCore. The agent gets placeholder AWS
# keys, and the box signs each request with your AWS credentials instead.
[egress.search]
destinations = ["$SEARCH_HOST"]
secret.ref = "aws://"

# Refused requests, for reading after the run.
[telemetry.records]
kind = "file"
destination = "$CONFIG_DIR/records.jsonl"
include = ["deny"]
""")
'''),
    code(r'''
POLICY_DW = Template("""
// Anything no permit matches is refused, and a forbid always beats a permit.
// Paths under the home directory are written as ~/...

// The model: Bedrock in one region.
@id("model_connect")
@description("Connect to Bedrock on 443.")
permit (principal, action == Box::Action::"net:connect", resource)
when { context.input.host == "$MODEL_HOST" && context.input.port == 443 };

@id("model_request")
@description("HTTPS requests to Bedrock.")
permit (principal, action == Box::Action::"http:request", resource)
when { context.input.host == "$MODEL_HOST" && context.input.port == 443 };

// Web search: the harness's web_search tool, through Amazon Bedrock AgentCore.
@id("search_connect")
@description("Connect to AgentCore on 443.")
permit (principal, action == Box::Action::"net:connect", resource)
when { context.input.host == "$SEARCH_HOST" && context.input.port == 443 };

@id("search_request")
@description("Web searches through AgentCore.")
permit (principal, action == Box::Action::"http:request", resource)
when {
  context.input.host == "$SEARCH_HOST" && context.input.port == 443 &&
  context.input.method == "POST" && context.input.path == "/web-search/web-search.v1/invoke"
};

// The grader: health checks and validation evaluations only. /final is for the notebook,
// outside the box, so no rule permits it.
@id("grader_connect")
@description("Connect to the local grader.")
permit (principal, action == Box::Action::"net:connect", resource)
when { context.input.host == "127.0.0.1" && context.input.port == $GRADER_PORT };

@id("grader_health")
@description("Check the grader is up.")
permit (principal, action == Box::Action::"http:request", resource)
when {
  context.input.host == "127.0.0.1" && context.input.port == $GRADER_PORT &&
  context.input.method == "GET" && context.input.path == "/health"
};

@id("grader_evaluate")
@description("Evaluate a policy on the validation seeds.")
permit (principal, action == Box::Action::"http:request", resource)
when {
  context.input.host == "127.0.0.1" && context.input.port == $GRADER_PORT &&
  context.input.method == "POST" && context.input.path == "/evaluate"
};

// The budget: refuse /evaluate once $BUDGET evaluations have succeeded (status 200).
// Refused or failed calls don't count.
@id("evaluation_budget")
@description("At most $BUDGET successful evaluations.")
forbid (principal, action == Box::Action::"http:request", resource)
when { context.input.host == "127.0.0.1" && context.input.port == $GRADER_PORT && context.input.path == "/evaluate" }
when temporal {
  exists (n: Long). (
    (count for (t: Timepoint). where (
      formerly within 24h (
        Box::Action::"http:request"::response{
          input.host: "127.0.0.1", input.port: $GRADER_PORT, input.path: "/evaluate", output.status: 200
        } && tp(t)
      )
    )) == n
    && n >= $BUDGET
  )
};

// The web, for the harness's web tools. GitHub is refused because the repo, criteria included,
// is public there.
@id("web_connect")
@description("Connect to any host on 443.")
permit (principal, action == Box::Action::"net:connect", resource)
when { context.input.port == 443 };

@id("web_request")
@description("HTTPS requests to any host.")
permit (principal, action == Box::Action::"http:request", resource)
when { context.input.port == 443 };

@id("no_github")
@description("No GitHub: the public repo holds the criteria.")
forbid (principal, action == Box::Action::"http:request", resource)
when {
  context.input.host == "github.com" || context.input.host == "api.github.com" ||
  context.input.host == "codeload.github.com" || context.input.host == "gist.github.com" ||
  context.input.host like "*.githubusercontent.com"
};

// Commands. The box's shell runs its own builtins (cat, ls, sed, curl, ...); the only host
// program it may start is Python, as [tool.python] in box.toml.
@id("shell_commands")
@description("Commands the box's shell implements itself.")
permit (principal, action == Box::Action::"shell:exec", resource);

@id("spawn_python")
@description("Start the venv's Python.")
permit (principal, action == Box::Action::"shell:spawn", resource)
when { context.input.program_path == "$PYTHON_PROGRAM" };

// Files through the shell: the workspace, plus reading agent.py and the rl code.
@id("workspace_read")
@description("Read the workspace.")
permit (principal, action == Box::Action::"fs:read", resource)
when { context.input.path == "$P_WORKSPACE" || context.input.path like "$P_WORKSPACE/*" };

@id("workspace_change")
@description("Write, move and delete in the workspace.")
permit (
  principal,
  action in [Box::Action::"fs:write", Box::Action::"fs:delete", Box::Action::"fs:move"],
  resource
)
when { context.input.path == "$P_WORKSPACE" || context.input.path like "$P_WORKSPACE/*" };

@id("code_read")
@description("Read the training code and agent.py.")
permit (principal, action == Box::Action::"fs:read", resource)
when { context.input.path like "$P_REPO/rl/*" || context.input.path == "$P_SCRIPT" };

@id("criteria_hidden")
@description("The success criteria and the example solution stay hidden.")
forbid (principal, action == Box::Action::"fs:read", resource)
when {
  context.input.path == "$P_REPO/rl/evaluate.py" ||
  context.input.path like "$P_REPO/rl/__pycache__/*" ||
  context.input.path == "$P_REPO/rl/rewards/examples/landing.py"
};
""")
'''),
    md("""
`write_box` fills in both templates for agent `n` of a run and writes them to
`<run>/box/agent-<n>/`, next to the agent's workspace `<run>/agent-<n>/`. The agent can't reach
its own box files. `budget` is how many sealed evaluations the agent gets, and `steps` caps each
training run.
"""),
    code("""
import sysconfig
from urllib.parse import urlparse

from rl.grader_client import grader_url

# Box keeps a socket in box_dir and macOS limits socket paths to 104 bytes, so keep it short.
BOX_ROOT = pathlib.Path.home() / ".autorl" / "box"
HOME = str(pathlib.Path.home())


def write_box(run, n, script="agent.py", budget=6, steps=300_000, target=0.90):
    run = pathlib.Path(run).resolve()
    workspace, config = run / f"agent-{n}", run / "box" / f"agent-{n}"
    # Box refuses to deny a path that doesn't exist, and rl/__pycache__ holds evaluate.py compiled.
    for path in (workspace / "rewards", workspace / "artifacts", workspace / ".tmp", config, BOX_ROOT,
                 REPO / "rl" / "__pycache__"):
        path.mkdir(parents=True, exist_ok=True)
    python = pathlib.Path(sys.executable)
    # A system Python lives under /usr, which Box won't grant, so grant its standard library only.
    python_base = pathlib.Path(sys.base_prefix).resolve()
    if python_base == pathlib.Path("/usr"):
        python_base = pathlib.Path(sysconfig.get_paths()["stdlib"])
    # On Linux the agent's process only loads C extensions from exec grants, and the system libraries
    # they link from the loader directory.
    linux_read, linux_exec = [], []
    if sys.platform == "linux":
        linux_read = [f"/usr/lib/{sysconfig.get_config_var('MULTIARCH')}", "/etc/ld.so.cache"]
        linux_exec = [sysconfig.get_config_var("DESTSHARED"), sysconfig.get_paths()["platlib"]]
    # Box policies spell paths under the home directory as ~/..., hence the P_ values.
    values = {
        "NAME": f"autorl-{run.name}-{n}", "BOX_DIR": BOX_ROOT / f"{run.name}-{n}",
        "PYTHON": python, "PYTHON_PROGRAM": str(python.resolve()).replace(HOME, "~", 1),
        "PYTHON_BASE": python_base, "VENV": sys.prefix,
        "LINUX_READ": "".join(f', "{path}"' for path in linux_read),
        "LINUX_EXEC": ", ".join(f'"{path}"' for path in linux_exec),
        "REPO": REPO, "P_REPO": str(REPO).replace(HOME, "~", 1),
        "WORKSPACE": workspace, "P_WORKSPACE": str(workspace).replace(HOME, "~", 1),
        "SCRIPT": run / script, "P_SCRIPT": str(run / script).replace(HOME, "~", 1), "CONFIG_DIR": config,
        "SEED": n, "BUDGET": budget, "TARGET": f"{target:.2f}", "STEPS": steps,
        "REGION": REGION, "MODEL_HOST": f"bedrock-runtime.{REGION}.amazonaws.com",
        "SEARCH_HOST": f"bedrock-agentcore.{REGION}.amazonaws.com",
        "GRADER_URL": grader_url(), "GRADER_PORT": urlparse(grader_url()).port,
    }
    (config / "box.toml").write_text(BOX_TOML.substitute(values))
    (config / "policy.dw").write_text(POLICY_DW.substitute(values))
    return config / "box.toml"


write_box(RUN, 1, budget=6, steps=300_000)
"""),
    md("""
## 5. Start it

The next cell starts the agent. First it mints a short-term Amazon Bedrock API key from your AWS
credentials. The agent never sees that key. Box handles the credentials: the agent's process only
gets a placeholder, and the box swaps in the real key on requests to Bedrock as they leave the
sandbox (`[egress.model]` in `box.toml`). So nothing the agent runs, and nothing it writes to a
file or a log, can leak the key, and the box only ever sends it to Bedrock.

Then the cell runs `box run --config <run>/box/agent-1/box.toml` in the background, with
the output going to `<run>/agent-1.log`, so you can carry on while the agent works. A run takes
about half an hour on an Apple MacBook Pro with an M3 Pro chip.
"""),
    code("""
from aws_bedrock_token_generator import provide_token

if LIVE:
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = provide_token(region=REGION)
    # box run --config <run>/box/agent-1/box.toml > <run>/agent-1.log 2>&1 &
    subprocess.Popen(["box", "run", "--config", f"{RUN}/box/agent-1/box.toml"], stdout=open(f"{RUN}/agent-1.log", "w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    print("agent started, logging to", f"{RUN}/agent-1.log")
"""),
    md("""
## 6. The grader

### 6.1 A misspecified reward

This reward is broken on purpose. An agent that writes its own reward judges progress by that
reward, but the number going up doesn't mean the task got done. Knowing whether the lander
actually landed takes a check the agent can't see or change: a sealed grader. Here's the reward.
It pays the lander for being close to a point just above the pad:

```python
# Rewards being close to a point just above the pad.
def reward(obs, action, next_obs, terminated, truncated, info):
    x, y, vx, vy, angle, ang_vel, leg1, leg2 = next_obs
    dist = ((x - 0.0) ** 2 + (y - 1.0) ** 2) ** 0.5
    return 1.0 - min(1.0, dist)
```

We trained it for 1,000,000 steps. This is what it learned:

![The naive hover policy holding the lander in the air above the pad without landing](images/hover.gif)

It hangs in the air above the pad until the episode times out. While training, we sent a
checkpoint to the grader every 50,000 steps. The grader measures what we actually want: the
share of episodes that end with the lander at rest, upright, on the pad, after a gentle
touchdown. That's the landing rate.

![Line chart over 1,000,000 training steps: the training reward climbs from about 58 to about 924
while the sealed landing rate stays at 0](images/gap.png)

Training reward per episode climbs from about 60 to over 900. The landing rate stays at zero
the whole time, far below the 0.90 target. The policy learned to do exactly what the reward pays
for.

This is a misspecified reward: it pays for the wrong thing, so the reward goes up and the task
doesn't get done. A person spots it in a few seconds of video. An agent working on its own needs
a check that is separate from its reward and that it can't change. That check is the grader.

### 6.2 Grading a policy

The grader is a small HTTP service on your machine, outside the agent's sandbox, and the first
cell of section 3 started it. It holds the success criteria in [`rl/evaluate.py`](../rl/evaluate.py).

To grade a policy, the grader flies it on the unmodified LunarLander environment, with the
environment's own reward and not the agent's, for a number of episodes. Each episode starts from
a fixed seed: 10,000 and up for the agent's evaluations, 20,000 and up for the final leaderboard.
Then it looks at how each episode ended:

- **Landed:** the lander came to rest with both legs down, moving slower than 0.3 in each
  direction, tilted less than 0.25 radians (about 14 degrees), and within 0.3 of the pad's
  centre.
- **Crashed:** the episode ended any other way before 1,000 steps: a crash, flying off the
  screen, or coming to rest without passing the landing test.
- **Timed out:** it was still flying after 1,000 steps.

It also measures `loiter_fraction`, the share of steps spent in the air over the pad barely
moving up or down. The policy passes when its landing rate is 0.90 or more.

This is an example of the response the agent sees when it calls `evaluate_policy`, here for the
hover policy:
"""),
    code("""
from rl.grader_client import ensure_grader, evaluate

ensure_grader()
evaluate("results/gap", n_episodes=20)
"""),
    md("""
Every episode times out and `loiter_fraction` is high: the numbers say "hovering" without anyone
watching the video.

## 7. Sealing the evaluation

An agent that's scored on a number will push that number up any way it can. If it could read
the criteria, it could aim at the exact thresholds instead of landing well. If it could change
the grader or evaluate as often as it liked, it could keep trying until a weak policy got lucky.
Sealing the evaluation means the agent can only get a score by sending a policy and reading the
numbers that come back, a limited number of times.

### 7.1 How the pieces fit

This is how the pieces fit together:

```
this notebook (host)
 |
 |-- grader: python -m rl.grader on 127.0.0.1:8899
 |     holds the success criteria (rl/evaluate.py)
 |     GET /health   POST /evaluate (validation seeds)   POST /final (test seeds)
 |
 `-- box run --config runs/<run>/box/agent-<n>/box.toml
       Strands Box checks files, programs and network requests against policy.dw
       `-- python runs/<run>/agent.py   (Strands harness agent on Amazon Bedrock)
             writes rewards/iter_<k>.py, trains into artifacts/iter_<k>/
             evaluate_policy -> POST /evaluate        allowed, up to the budget
             rl/evaluate.py, /final, GitHub           refused
```

The judgement stays out of the agent's reach in five ways:

1. The criteria live only in the grader process. The agent sends weights and gets numbers back.
2. The agent's evaluations use validation episodes (seeds 10,000 and up). The final leaderboard
   uses test episodes (seeds 20,000 and up) through `/final`, which only the notebook can call.
3. Training checks every reward the agent writes before it runs (`rl/guard.py`): only
   arithmetic and a few math functions, no imports, no file or network access.
4. The box stops the agent reading the criteria, calling `/final`, or fetching the public copy
   of this repo.
5. The box refuses evaluations beyond the budget.

### 7.2 The box

These are the files `write_box` wrote for your agent:
"""),
    code("""
from IPython.display import Code

Code(filename=f"{RUN}/box/agent-1/box.toml", language="toml")
"""),
    code('Code(filename=f"{RUN}/box/agent-1/policy.dw", language="text")'),
    md("""
Things to look for:

- `[agent.filesystem]` grants reading the workspace, `agent.py` and the `rl` package, and denies
  `rl/evaluate.py`, its compiled copy in `rl/__pycache__`, and the example solution. The agent
  can see how training and the reward checks work. On Linux it also grants the system libraries
  and lets Python load its C extensions (`exec`).
- `[limits]` lets one shell call run for up to 30 minutes, since a training run takes minutes.
- `[tool.python]` is the venv's Python. A bare `python` inside the box is a restricted
  interpreter, so the agent runs training with the full path it's given.
- `evaluation_budget` counts successful responses from `/evaluate` and refuses the next request
  once the count reaches the budget. Refused calls don't count. The stop rule normally ends the
  run first; this rule holds even if the agent's own process doesn't.
- `[egress.search]` and `search_request` let the harness's `web_search` tool search the web through
  Amazon Bedrock AgentCore. The box signs those requests with your AWS credentials, so the agent
  never holds them.
- `no_github` blocks GitHub, because this repo, criteria included, is public there. Reading
  the criteria wouldn't change a grade anyway, since the agent only ever gets numbers back from
  episodes it hasn't trained on, but there's no reason to make it easy.

### 7.3 Try to get out

To check the seal, we ran a script inside a box with the same config as the agent's and a budget
of 2. It did what a curious agent might try: read the criteria, `cat` them through the box's
shell, fetch the copy on GitHub, call `/final`, and evaluate once more than the budget allows.
This is what happened:
"""),
    code('print(open("results/seal_check.txt").read())'),
    md("""
Line by line:

- **read rl/evaluate.py:** the script's own Python process tried to open the file. Box started
  that process without access to it (`deny` in `[agent.filesystem]`), so the operating system
  refused it.
- **cat rl/evaluate.py in the shell:** the box's shell checked the read against `policy.dw`,
  and the `criteria_hidden` rule refused it.
- **fetch the GitHub copy:** the request went through the box's network gate, and the
  `no_github` rule refused it.
- **grade on the test seeds (/final):** no rule permits `/final`, so the box refused the request
  before it reached the grader (`default-deny`).
- **evaluations 1 and 2:** allowed, and the grader scored a policy of all zeros at 0.0.
- **evaluation 3:** the `evaluation_budget` rule counted two successful evaluations and refused
  the third.
"""),
    md("""
## 8. What the agent did

### 8.1 Your run

Here's where the agent you started in section 5 has got to. Rerun the cell to refresh it.
"""),
    code("""
if LIVE:
    !tail -n 15 {RUN}/agent-1.log
    !cat {RUN}/agent-1/artifacts/eval_log.jsonl 2>/dev/null
"""),
    md("""
### 8.2 A saved run

This is a saved run of the same code on a laptop, with the same settings as yours: a budget of 6
sealed evaluations and a 300,000-step ceiling. These are its sealed evaluations:
"""),
    code("""
import json

from IPython.display import Code

SAVED = "results/box1/agent-1"
rows = [json.loads(line) for line in open(f"{SAVED}/artifacts/eval_log.jsonl")]
for row in rows:
    print(f"{row['run_dir']:28} {row['n_episodes']:3} episodes  landing {row['landing_rate']:.3f}"
          f"  crash {row['crash_rate']:.3f}  timeout {row['timeout_rate']:.3f}  passed {row['passed']}")
"""),
    md("""
Its first two policies crashed. The agent worked out from the numbers that its per-step time
penalty made diving the cheapest way to end an episode. Its third policy went the other way:
every episode timed out and `loiter_fraction` was 0.85, so the lander spent most of each episode
hovering over the pad. That's the same failure as the misspecified reward in section 6, and the
agent spotted it from the grader's numbers alone.

With three evaluations left, it stopped spending them on guesses. It flew candidates on the plain
environment with its own short checks, and worked through 27 more rewards that way. It hit
diving, freezing at the starting altitude and landing off-centre along the way, and found that
the training seed mattered a lot. Its fourth evaluation scored 0.90, and the stop rule ended the run
with two evaluations unused.

The first reward it wrote:
"""),
    code("""
summary = json.load(open(f"{SAVED}/{rows[0]['run_dir']}/train_summary.json"))
Code(filename=f"{SAVED}/{summary['reward_path']}")
"""),
    md("The reward that passed, and how the run ended:"),
    code("""
summary = json.load(open(f"{SAVED}/{rows[-1]['run_dir']}/train_summary.json"))
display(Code(filename=f"{SAVED}/{summary['reward_path']}"))
print("\\n".join(pathlib.Path("results/box1/agent-1.log").read_text().splitlines()[-6:]))
"""),
    md("""
And here it lands:

![The single-agent saved run's passing policy bringing the lander down between the flags](images/box1.gif)
"""),
    md("""
### 8.3 Your run, when it's finished

The same views of your own agent: its sealed evaluations, the reward behind its best result, and
a GIF of that policy flying.
"""),
    code("""
import json

from IPython.display import Code, Image

from rl.render import record

MINE = f"{RUN}/agent-1"
log = pathlib.Path(f"{MINE}/artifacts/eval_log.jsonl")
mine = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
for row in mine:
    print(f"{row['run_dir']:28} {row['n_episodes']:3} episodes  landing {row['landing_rate']:.3f}"
          f"  crash {row['crash_rate']:.3f}  timeout {row['timeout_rate']:.3f}  passed {row['passed']}")
if mine:
    best = max(mine, key=lambda row: row["landing_rate"])
    summary = json.load(open(f"{MINE}/{best['run_dir']}/train_summary.json"))
    display(Code(filename=f"{MINE}/{summary['reward_path']}"))
    display(Image(record(f"{MINE}/{best['run_dir']}", f"{RUN}/best.gif")))
else:
    print("no sealed evaluations yet")
"""),
    md("""
## 9. Parallel agents

One run is one sample. The same agent with the same brief can pass or fall short, depending on
its early choices. Running several agents side by side, each with a fresh context (its own
folder, memory, budget and training seed), explores several paths at once, and the grader picks
the winner.

The next cell makes a run folder with three agents, each with its own box, and starts them the
same way as in section 5, next to the one you started earlier, so expect your machine to be busy.
"""),
    code("""
import shutil

# Box names hold at most 32 bytes, and each agent's includes this folder's name.
PAR = f"{RUN}-par"
os.makedirs(PAR)
shutil.copy(f"{RUN}/agent.py", PAR)
for n in (1, 2, 3):
    write_box(PAR, n, budget=6, steps=300_000)
if LIVE:
    for n in (1, 2, 3):
        subprocess.Popen(["box", "run", "--config", f"{PAR}/box/agent-{n}/box.toml"],
                         stdout=open(f"{PAR}/agent-{n}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    print("three agents started in", PAR)
"""),
    md("""
This is a saved run of the same cells on a laptop: three agents, each with a budget of 6.
Their sealed landing rate after each evaluation:

![Line chart of the sealed landing rate at each evaluation for three agents: agent-1 reaches 0.90 at
evaluation 5, agent-3 reaches 0.90 at evaluation 6, and agent-2 peaks at 0.88 at evaluation 4 and
ends at 0.66](images/par3.png)

Agents 1 and 3 passed, on their fifth and sixth evaluations. Agent 2 used all six, and its best
was 0.88.

The leaderboard takes each agent's best policy, going by its latest validation result for each
policy, and grades that one policy once on the test seeds through `/final`, which no agent can
reach.
"""),
    code("""
import json

from rl.grader_client import GraderError, evaluate

# Use your own parallel run folder, PAR, here once its agents have finished.
PARALLEL = "results/par3"

board = []
for workspace in sorted(pathlib.Path(PARALLEL).glob("agent-*/")):
    log, latest = workspace / "artifacts" / "eval_log.jsonl", {}
    # The agent can write to its eval log, so skip anything malformed.
    for line in log.read_text().splitlines() if log.exists() else []:
        try:
            row = json.loads(line)
            latest[str(row["run_dir"])] = float(row["landing_rate"])
        except (ValueError, KeyError, TypeError):
            continue
    if not latest:
        board.append((workspace.name, "-", None, None))
        continue
    best = max(latest, key=latest.get)
    policy = (workspace / best).resolve()
    try:
        if not policy.is_relative_to(workspace.resolve()):
            raise GraderError("outside the agent's workspace")
        test = evaluate(policy, 100, endpoint="final")["landing_rate"]
    except GraderError:
        test = None
    board.append((workspace.name, best, latest[best], test))

board.sort(key=lambda row: -1 if row[3] is None else row[3], reverse=True)
for agent, policy, validation, test in board:
    scores = "no graded policy" if test is None else f"validation {validation:.2f}  test {test:.2f}"
    print(f"{agent:10} {policy:28} {scores}")
"""),
    md("""
Why grade again on new seeds? Each agent's results come from 50 episodes, a small sample that can
make a policy look better or worse than it is, and picking the best of many results flatters it
again. These are all the agents' evaluations:
"""),
    code("""
for n in (1, 2, 3):
    for line in open(f"results/par3/agent-{n}/artifacts/eval_log.jsonl"):
        row = json.loads(line)
        print(f"agent-{n}  {row['run_dir']:28} {row['n_episodes']:3} episodes  landing {row['landing_rate']:.3f}")
    print()
"""),
    md("""
Agent 2 never passed during its run: its best policy landed on 0.88 of 50 validation episodes.
Grading each chosen policy once, on seeds that nobody trained or selected on, gives a score you
can trust.

The winner:

![The parallel run's winning policy, from agent-3, bringing the lander down between the flags](images/par3.gif)

## 10. Wrap-up

Agents can run large parts of the experiment cycle themselves. In this notebook the agent wrote
the rewards, trained the policies, read the grades, worked out what went wrong and tried again,
the loop a person would otherwise drive by hand. That lets us run more experiments, and run them
at a bigger scale, than we could by hand.

Nothing in the loop is specific to LunarLander. The method, the tool, the stop rule and the box
stay the same; for another task you change the objective and the grader's criteria. Any problem
with a simulator and a measurable success test fits.

Parallel agents explore different paths at once, each from a fresh context, which raises the
chance that one of them finds a good solution. One grader scores them all on the same test
seeds.

The judge and the limits around the agent are what make its results trustworthy: criteria in a
separate process, held-out test seeds, checked code, a stop rule in a hook, and a budget and step
ceiling that the box and the trainer enforce. Agents aren't reliable at keeping to a budget
they've only been told about.

To stop any agents that are still running, set `STOP = True` in the next cell and run it. Each
box takes up to half a minute to exit. It's off by default so that running every cell at once
doesn't stop the agents you just started.
"""),
    code("""
STOP = False
if STOP:
    !pkill -f "box run --config {RUN}"
    !pkill -f "box run --config {PAR}"
"""),
    md("""
To delete every run folder and the box state once you no longer need them, set `DELETE = True`
in the next cell and run it.
"""),
    code("""
DELETE = False
if DELETE:
    !rm -rf runs ~/.autorl/box
"""),
]

def build(cells) -> nbf.NotebookNode:
    nb = new_notebook(cells=cells)
    # nbformat gives each cell a random id; fixed ids keep rebuilds identical.
    for index, cell in enumerate(nb.cells):
        cell.id = f"cell-{index:02d}"
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata.language_info = {"name": "python"}
    return nb


def main() -> None:
    nbf.write(build(CELLS), str(NOTEBOOK))
    print("wrote", NOTEBOOK.name)


if __name__ == "__main__":
    main()
