# CI Path Filters Configuration

## Overview

GitHub Actions workflows are now configured to **skip unnecessary builds** when only documentation files are changed. This saves CI/CD resources and time.

## What Changed

### Before
- **Any** push to main/master triggered all CI workflows
- Documentation changes triggered full Python builds and tests
- Wasted CI minutes and resources

### After
- Documentation-only changes **skip** code CI workflows
- Code changes still trigger all necessary workflows
- More efficient resource usage

## Workflows Affected

### 1. `release-python.yml` - Python Library Release
**Skips when only these files change:**
- `docs/**` - All documentation files
- `*.md` - All markdown files in root
- `mkdocs.yml` - MkDocs configuration
- `.github/workflows/docs.yml` - Docs workflow
- `DOCS_*.md` - Documentation guides
- `TOC_*.md` - Table of contents guides
- `COLOR_*.md` - Color scheme guides

### 2. `release-python-cli.yml` - Python CLI Release
**Same skip conditions as above**

### 3. `docs.yml` - Documentation Deployment
**Always runs** (no path filters) - This is intentional!

### 4. `labeler.yml` - GitHub Labels
Already has specific path filters (unchanged)

## How Path Filters Work

### Example 1: Documentation-Only Change

```bash
# Scenario: Update tutorial.md
git add docs/tutorial.md
git commit -m "docs: improve tutorial examples"
git push

# Result:
✅ docs.yml - RUNS (deploys documentation)
⏭️ release-python.yml - SKIPPED
⏭️ release-python-cli.yml - SKIPPED
```

### Example 2: Code Change

```bash
# Scenario: Update Python code
git add deepchopper/model.py
git commit -m "feat: improve model accuracy"
git push

# Result:
✅ docs.yml - RUNS (always runs on main)
✅ release-python.yml - RUNS (builds and tests)
✅ release-python-cli.yml - RUNS (builds CLI)
```

### Example 3: Mixed Changes

```bash
# Scenario: Update both code and docs
git add deepchopper/model.py docs/tutorial.md
git commit -m "feat: improve model and update docs"
git push

# Result:
✅ docs.yml - RUNS
✅ release-python.yml - RUNS (code changed)
✅ release-python-cli.yml - RUNS (code changed)
```

### Example 4: README Update

```bash
# Scenario: Update README.md
git add README.md
git commit -m "docs: update README badges"
git push

# Result:
✅ docs.yml - RUNS
⏭️ release-python.yml - SKIPPED (only .md changed)
⏭️ release-python-cli.yml - SKIPPED (only .md changed)
```

## Path Patterns Explained

### `docs/**`
Matches all files in the `docs/` directory and subdirectories:
- `docs/index.md`
- `docs/tutorial.md`
- `docs/stylesheets/extra.css`
- `docs/overrides/main.html`

### `*.md`
Matches all markdown files in the **root** directory:
- `README.md`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `DOCS_DEPLOYMENT.md`
- etc.

**Does NOT match:**
- `deepchopper/README.md` (not in root)
- `tests/fixtures/test.md` (not in root)

### `mkdocs.yml`
Matches only the MkDocs configuration file in root

### `.github/workflows/docs.yml`
Matches only the docs workflow file

### `DOCS_*.md`, `TOC_*.md`, `COLOR_*.md`
Matches documentation guide files:
- `DOCS_DEPLOYMENT.md`
- `DOCS_QUICK_REFERENCE.md`
- `TOC_IMPROVEMENTS.md`
- `COLOR_SCHEME_UPDATE.md`

## Special Cases

### Tags
**Path filters are ignored for tags:**
```bash
git tag v1.0.0
git push --tags

# Result: ALL workflows run regardless of what changed
✅ docs.yml - RUNS
✅ release-python.yml - RUNS (creates release)
✅ release-python-cli.yml - RUNS (creates release)
```

This is intentional - releases should always build everything.

### Manual Dispatch
**Path filters are ignored for manual triggers:**
```bash
# Via GitHub UI: Actions → Workflow → Run workflow

# Result: Workflow runs regardless of path filters
```

### Pull Requests
Path filters apply to PRs:
```bash
# PR with only doc changes
✅ docs.yml - RUNS (verifies docs build)
⏭️ release-python.yml - SKIPPED
⏭️ release-python-cli.yml - SKIPPED
```

## Benefits

### 1. Faster Feedback
- Doc changes get immediate feedback from docs workflow
- Don't have to wait for unnecessary Python builds

### 2. Resource Savings
- Typical Python build: ~10-20 minutes
- CI minutes saved: ~20-40 minutes per doc-only push
- Cost savings for GitHub Actions

### 3. Cleaner CI History
- Less clutter in Actions tab
- Easier to find relevant build results

### 4. Better Development Flow
- Writers can update docs without triggering builds
- Developers see builds only for code changes

## Configuration Details

### release-python.yml

```yaml
on:
  push:
    branches:
      - main
      - master
    tags:
      - "*"
    paths-ignore:
      - 'docs/**'
      - '*.md'
      - 'mkdocs.yml'
      - '.github/workflows/docs.yml'
      - 'DOCS_*.md'
      - 'TOC_*.md'
      - 'COLOR_*.md'
  pull_request:
    branches:
      - main
      - master
    paths-ignore:
      - 'docs/**'
      - '*.md'
      - 'mkdocs.yml'
      - '.github/workflows/docs.yml'
      - 'DOCS_*.md'
      - 'TOC_*.md'
      - 'COLOR_*.md'
  workflow_dispatch:
```

### release-python-cli.yml

Same configuration as above.

### docs.yml

**No path filters** - Always runs to ensure documentation stays up to date.

## Testing Path Filters

### Test Locally

You can test if your changes would trigger workflows:

```bash
# Install gh CLI if not already installed
# gh CLI: https://cli.github.com/

# Check what would trigger
gh api repos/:owner/:repo/actions/workflows/release-python.yml
```

### Test on GitHub

1. Create a branch with doc-only changes
2. Push to GitHub
3. Check Actions tab - should only see docs workflow

## Troubleshooting

### Workflow Not Skipping

**Problem:** Changed only docs but workflow still ran

**Possible causes:**
1. Mixed changes (code + docs)
2. Push included a tag
3. Manual workflow dispatch
4. Path pattern didn't match

**Solution:** Check what files changed:
```bash
git log -1 --name-only
```

### Workflow Skipped When Shouldn't

**Problem:** Changed code but workflow didn't run

**Possible causes:**
1. Path pattern too broad
2. File matched ignore pattern

**Solution:** Check ignored paths match your change

### Need to Force Run

**Solution:** Use manual workflow dispatch:
1. Go to Actions tab
2. Select workflow
3. Click "Run workflow"
4. Choose branch
5. Click "Run workflow" button

## Best Practices

### 1. Commit Docs Separately

```bash
# Good: Separate commits
git add docs/
git commit -m "docs: update tutorial"
git push

git add deepchopper/
git commit -m "feat: new feature"
git push
```

```bash
# Less optimal: Mixed commit
git add docs/ deepchopper/
git commit -m "feat: new feature with docs"
git push
# ↑ This triggers all workflows
```

### 2. Update README Separately

README updates often don't need CI:
```bash
git add README.md
git commit -m "docs: update installation instructions"
git push
# Only docs workflow runs ✅
```

### 3. Use Conventional Commits

- `docs:` - Documentation changes
- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code refactoring

Helps identify what changed quickly.

## Important Notes

### Tags Override Path Filters

When you create a release tag, **all workflows run** regardless of what changed. This is by design - releases should always be fully tested.

```bash
git tag v1.0.0
git push --tags
# All workflows run even if only docs changed
```

### Documentation Workflow Always Runs

The `docs.yml` workflow has **no path filters** and always runs on pushes to main. This ensures:
- Documentation stays synchronized
- Changes are immediately deployed
- Version selector stays updated

### Negation Patterns

We use `paths-ignore` instead of `paths` because:
- **paths-ignore**: Skip if ONLY these files change
- **paths**: Run ONLY if these files change

`paths-ignore` is better because:
- Default is to run (safer)
- Explicitly lists exceptions
- Easier to understand

## Summary

**Before:** Every push triggered all CI workflows
**After:** Only relevant workflows run based on file changes

**Benefits:**
- ⚡ Faster feedback for doc changes
- 💰 Saves CI minutes and costs
- 🎯 Cleaner CI history
- 🔧 Better developer experience

**Affected Workflows:**
- ✅ `release-python.yml` - Skips for doc-only changes
- ✅ `release-python-cli.yml` - Skips for doc-only changes
- ℹ️ `docs.yml` - Always runs (no filters)
- ℹ️ `labeler.yml` - Already has specific filters

**Path Patterns:**
- `docs/**` - All doc files
- `*.md` - Root markdown files
- `mkdocs.yml` - MkDocs config
- `DOCS_*.md`, `TOC_*.md`, `COLOR_*.md` - Doc guides

The configuration is production-ready and will save significant CI resources! 🎉
