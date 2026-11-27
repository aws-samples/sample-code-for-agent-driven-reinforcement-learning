# Agents that train themselves: autonomous RL on AWS

In this workshop an AI agent runs a reinforcement learning project by itself. You give it one
goal, "land the lunar lander gently on the pad". The agent writes a reward function, trains a
policy, has it graded by an evaluator it can't see, works out what went wrong from the numbers,
and tries again until it lands.

An agent that optimizes a reward will take any shortcut the reward allows, so the agent is judged
by something it can't change. The success criteria live in a separate grader process, the agent
runs inside a [Strands Box](https://strandsagents.com/docs/user-guide/box/) sandbox that refuses
access to them, its evaluation budget is enforced by the box, and the winner of a parallel run is
graded once more on held-out test episodes.

The workshop is one notebook, `notebooks/workshop_01_agents_that_train_themselves.ipynb`, and all
of the agent's code is in it. The notebook builds the agent step by step, puts it in a box, starts
it, and then covers a misspecified reward, the grader, sealing the evaluation, one agent's run,
and three agents in parallel.

## How it works

```
the notebook (host)
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

The agent is the [Strands harness](https://strandsagents.com/docs/user-guide/harness/) default
agent with one extra tool, `evaluate_policy`, which sends a trained policy's weights to the grader
and returns metrics, and one hook, `StopRule`, which ends the run when the target is met, the
budget is used up or the results stop improving. The notebook writes the agent's program,
`agent.py`, and each agent's `box.toml` and `policy.dw` into the run folder under `runs/`. Each
agent works in its own folder there, with its own memory, sessions and evaluation budget.

## What's in the repo

| Path | What it is |
| --- | --- |
| `notebooks/` | The workshop notebooks, `_build.py`, which generates them, and the images they show |
| `rl/env.py`, `rl/guard.py`, `rl/train.py` | LunarLander with an agent-written reward, the checks on that reward, and PPO training |
| `rl/evaluate.py`, `rl/grader.py` | The success criteria and the grader service that holds them |
| `rl/numpy_policy.py`, `rl/grader_client.py` | The policy forward pass in numpy (used by the grader) and the grader's HTTP client |
| `rl/experiment.py`, `rl/plot.py`, `rl/render.py` | Training with periodic grading, the plots, and GIFs of a policy flying |
| `results/` | Saved results from real runs, used by the notebook |
| `tests/` | Tests for the numpy policy and the grader |

## Where it runs

You can run it on your own Mac or Linux machine, or on an EC2 instance. On Linux you need an
x86_64 machine, because Box2D 2.3.10 has no wheels for Linux on Arm. On EC2, use an x86_64
instance (for example from the c7i family) and set it up like any Linux machine; the instance
role's credentials can mint the Bedrock API key.

Ubuntu 24.04 only lets programs with an AppArmor `userns` rule create user namespaces, and Box's
launcher needs one. With Box in `~/.local/bin`, this profile grants it to the launcher alone:

```bash
sudo tee /etc/apparmor.d/strands-box <<EOA
abi <abi/4.0>,
include <tunables/global>

profile strands-box $HOME/.local/bin/strands-box-contain-trampoline flags=(unconfined) {
  userns,
}
EOA
sudo apparmor_parser -r /etc/apparmor.d/strands-box
```

See [Box's Linux requirements](https://strandsagents.com/docs/user-guide/box/reference/linux-requirements/)
for other distributions.

## Prerequisites

- Python 3.11, 3.12 or 3.13 (Box2D has no wheels for 3.14).
- Strands Box 0.3.1 or later: the `box` executable and its two helpers, together, on your `PATH`
  or in `~/.local/bin` (the notebook looks there too, since editor-started kernels often lack your
  shell's `PATH`). See https://strandsagents.com/docs/user-guide/box/.
- An AWS account with Amazon Bedrock in `us-west-2`. The agent uses the Strands harness's default
  models, a main model and a smaller one for memory and web page summaries, through global
  inference profiles.
- AWS credentials in your environment (a profile, SSO or an instance role) that can call those
  models (`bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`), use Bedrock API keys
  (`bedrock:CallWithBearerToken`) and search the web with Amazon Bedrock AgentCore
  (`bedrock-agentcore:InvokeWebSearch`). The notebook turns them into a short-term Bedrock API key
  for the box. If the account hasn't used a model before, the first call subscribes the account to
  it through AWS Marketplace, which also needs `aws-marketplace:ViewSubscriptions` and
  `aws-marketplace:Subscribe`.

Without Box or Bedrock access you can still read and run the notebook: set `LIVE = False` in its
first code cell, and it uses the saved results in place of live runs.

## Setup

```sh
git clone https://github.com/aws-samples/sample-code-for-agent-driven-reinforcement-learning.git
cd sample-code-for-agent-driven-reinforcement-learning
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[notebooks]'
```

Open the notebook in VS Code or Jupyter with the `.venv` interpreter as the kernel, and run it
from the top. To run Jupyter from the terminal:

```sh
jupyter lab notebooks/workshop_01_agents_that_train_themselves.ipynb
```

## Settings

The notebook's settings are plain values in its code cells: `LIVE` and `REGION` in the
first code cell of section 3, and each run's budget and step ceiling where it calls `write_box`.
The grader listens on `127.0.0.1:8899`.

## Saved results

`results/` holds a small set of files from real boxed runs on a laptop, laid out like a run
folder:

- `results/gap/`: the naive hover reward trained for 1,000,000 steps, with the grader's landing
  rate every 50,000 steps.
- `results/box1/`: one agent with a budget of 6 and a 300,000-step ceiling. It passed on its
  fourth evaluation with a landing rate of 0.90.
- `results/par3/`: three agents in parallel with a budget of 6. On the test seeds
  their best policies scored 0.96, 0.95 and 0.94.
- `results/seal_check.txt`: the output of the seal check.

Each folder keeps the eval logs, rewards and policy weights the notebook needs (and `box1` its
agent's log), so the notebook's leaderboard can regrade the parallel run.

## Cleanup

The last cells of the notebook stop the agents and delete the run folders and the box state. Stop
the grader with `kill $(lsof -ti tcp:8899 -sTCP:LISTEN)`. To clean up by hand:

```sh
pkill -f "box run --config runs/"
rm -rf runs ~/.autorl/box
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

MIT-0. See [LICENSE](LICENSE).
