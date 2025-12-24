# Logging Format Convention

All log messages use a **bracketed prefix tag** to indicate the message type, followed by the message content. This provides consistent, scannable output.

## Tag Types

| Tag | Purpose | Example |
|-----|---------|---------|
| `[info]` | General status updates, progress info | `[info] computing exact Z(β) sequence...` |
| `[trial XX/YY]` | Per-trial progress during experiments | `[trial 03/15] time=2.4s \| step err=1.74e-02` |
| `[summary]` | Aggregate results after a batch/sweep | `[summary] C=50 \| geom err: 2.12e+00 \| time=0.6min` |
| `[done]` | Task completion, file saves | `[done] saved plot -> ./plots/fig.png` |
| `[error]` | Error conditions (if applicable) | `[error] ran out of einsum indices` |
| `[result]` | Final computed values | `[result] exact Z = 1.004537e+00` |

## Format Rules

1. **Tag at start**: Always begin with `[tag]` followed by a space
2. **Lowercase tags**: Use lowercase for consistency (`[info]` not `[INFO]`)
3. **Pipe separators**: Use `|` to separate multiple metrics on one line
4. **Scientific notation**: Use `.2e` or `.3e` format for floating-point values
5. **Progress counters**: Format as `XX/YY` with zero-padded indices (`01/15`)
6. **Timing**: Include `time=X.Xs` or `time=X.Xmin` for duration tracking

## Example Output Block

```
[info] created tensor network for experiment
[info] computing exact Z(beta) for Linear schedule...
[info] exact Z at beta=1.0: Z_true = 1.004537e+00

[info] Linear | C = 50 mixing steps per beta step
[trial 01/15] time=2.5s | step err @ β=0.5 -> 1.21e-02 | Z(1) log err -> 2.48e+00
[trial 02/15] time=2.4s | step err @ β=0.5 -> 1.08e-02 | Z(1) log err -> 2.44e+00
...
[summary] C=50 | step geom err (last): 1.48e-02 | Z(1) geom err: 2.12e+00 | time=0.6min

[done] saved plot -> ./plots/ais_comparison.png
```

