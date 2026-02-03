# topic prune

The `topic prune` command removes branches or depth levels from topic graphs, reducing complexity or isolating specific subtrees for dataset generation.

!!! warning "Graph Files Only"
    Pruning operates on JSON graph files only. JSONL tree files are not supported.

## Basic Usage

Always preview changes with `--dry-run` first:

```bash title="Preview before pruning"
deepfabric topic prune topics.json --level 2 --dry-run
```

Then apply the prune:

```bash title="Prune and save"
deepfabric topic prune topics.json --level 2 -o pruned_topics.json
```

## Pruning Modes

Exactly one pruning mode must be specified per invocation.

### By Level

Remove all nodes below a given depth:

```bash title="Level-based pruning"
# Keep only root and first-level children
deepfabric topic prune topics.json --level 1 -o shallow.json

# Keep two levels of depth
deepfabric topic prune topics.json --level 2 -o medium.json
```

Level 0 keeps only the root node. Level 1 keeps the root and its direct children. Higher levels preserve more of the hierarchy.

### By UUID

Remove a specific node and its entire subtree:

```bash title="UUID-based pruning"
deepfabric topic prune topics.json --uuid abc-123-def -o pruned.json
```

!!! tip "Finding UUIDs"
    Use `deepfabric topic inspect topics.json --level 1 --uuid` to discover node UUIDs. See [topic inspect](topic-inspect.md).

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--level N` | `-l` | Remove all nodes below depth N |
| `--uuid ID` | `-u` | Remove a node and its subtree by UUID |
| `--output PATH` | `-o` | Output file path (default: auto-generated from input name) |
| `--force` | `-f` | Overwrite the input file instead of creating a new one |
| `--dry-run` | | Preview what would be removed without writing files |

## Safety Features

Pruning is designed to be safe by default:

:material-shield-check: **Dry run first**
:   Use `--dry-run` to see exactly what will be removed and what will remain before committing.

:material-file-plus: **New file by default**
:   Without `--force`, pruning writes to a new file (auto-named or specified with `-o`), leaving the original intact.

:material-shield-lock: **Root protection**
:   The root node cannot be removed. Attempting to prune the root by UUID raises an error.

!!! danger "Force Mode"
    The `--force` flag overwrites the original file. Use with caution — there is no undo.

## Dry Run Output

The `--dry-run` flag shows a preview of the pruning operation:

```bash title="Dry run example"
deepfabric topic prune topics.json --level 1 --dry-run
```

Output includes:

- Current graph size (total nodes)
- Number of nodes that would be removed
- Number of nodes that would remain
- No files are written

## Workflow

A typical prune workflow combines `inspect` and `prune`:

```bash title="Inspect, prune, verify"
# 1. Explore the structure
deepfabric topic inspect topics.json --level 1 --expand

# 2. Find the UUID of a branch to remove
deepfabric topic inspect topics.json --level 1 --uuid

# 3. Preview the removal
deepfabric topic prune topics.json --uuid abc-123-def --dry-run

# 4. Prune to a new file
deepfabric topic prune topics.json --uuid abc-123-def -o refined.json

# 5. Verify the result
deepfabric topic inspect refined.json --all
```

## Auto-Generated Output Paths

When no `-o` is specified and `--force` is not set, the output file is derived from the input:

| Input | Mode | Auto-generated Output |
|-------|------|-----------------------|
| `topics.json` | `--level 2` | `topics_pruned_level2.json` |
| `topics.json` | `--uuid abc` | `topics_pruned.json` |
