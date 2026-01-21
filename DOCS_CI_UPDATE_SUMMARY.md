# Documentation CI Update Summary

## Problem
The documentation CI workflow (`.github/workflows/docs.yml`) was only deploying the "latest" version without creating versioned documentation (e.g., 1.3.1, 1.3.0, etc.). Additionally, it only triggered on tags with the format `v*.*.*` but not `py-cli-v*.*.*`.

## Solution
Updated the workflow to:
1. **Support both tag formats**: `v*.*.*` and `py-cli-v*.*.*`
2. **Deploy versioned documentation**: Each tag now deploys both a versioned copy and updates "latest"

## Changes Made

### 1. Trigger Pattern Update (Line 5-7)
**Before:**
```yaml
on:
  push:
    tags:
      - "v*.*.*"
```

**After:**
```yaml
on:
  push:
    tags:
      - "v*.*.*"
      - "py-cli-v*.*.*"
```

### 2. Version Extraction Logic (Line 77-95)
**Before:**
```bash
if [[ $GITHUB_REF == refs/tags/v* ]]; then
  VERSION=${GITHUB_REF#refs/tags/v}
  mike deploy --push --update-aliases $VERSION latest
  mike set-default --push latest
else
  mike deploy --push dev
fi
```

**After:**
```bash
if [[ $GITHUB_REF == refs/tags/v* ]] || [[ $GITHUB_REF == refs/tags/py-cli-v* ]]; then
  # Extract version number from tag
  TAG=${GITHUB_REF#refs/tags/}
  if [[ $TAG == py-cli-v* ]]; then
    VERSION=${TAG#py-cli-v}
  else
    VERSION=${TAG#v}
  fi
  echo "Deploying version: $VERSION"
  mike deploy --push --update-aliases $VERSION latest
  mike set-default --push latest
else
  echo "Deploying dev version"
  mike deploy --push dev
fi
```

## How It Works Now

### When you push a tag:

#### Tag: `py-cli-v1.3.1`
1. Workflow triggers automatically
2. Extracts version: `1.3.1`
3. Deploys documentation to:
   - `https://ylab-hi.github.io/DeepChopper/1.3.1/` (versioned)
   - `https://ylab-hi.github.io/DeepChopper/latest/` (latest alias)
4. Sets `latest` as the default version

#### Tag: `v1.3.0`
1. Workflow triggers automatically
2. Extracts version: `1.3.0`
3. Deploys documentation to:
   - `https://ylab-hi.github.io/DeepChopper/1.3.0/` (versioned)
   - `https://ylab-hi.github.io/DeepChopper/latest/` (latest alias)
4. Sets `latest` as the default version

### When you push to main branch:
1. Workflow triggers automatically
2. Deploys documentation to:
   - `https://ylab-hi.github.io/DeepChopper/dev/` (development version)

## Version Selector

Users will now see a version selector in the documentation that allows them to switch between:
- **latest** (points to the most recent tagged version)
- **dev** (development/bleeding edge from main branch)
- **1.3.1** (specific version)
- **1.3.0** (specific version)
- **1.2.9** (specific version)
- etc.

## Testing the Changes

After pushing the updated workflow and the `py-cli-v1.3.1` tag:

1. **Check GitHub Actions**: 
   - Go to https://github.com/ylab-hi/DeepChopper/actions
   - Verify the "Deploy Documentation" workflow ran successfully

2. **Verify Documentation Deployment**:
   - Visit https://ylab-hi.github.io/DeepChopper/
   - Check that the version selector appears
   - Verify you can access:
     - https://ylab-hi.github.io/DeepChopper/latest/
     - https://ylab-hi.github.io/DeepChopper/1.3.1/
     - https://ylab-hi.github.io/DeepChopper/dev/

3. **Check gh-pages branch**:
   ```bash
   git fetch origin
   git checkout gh-pages
   ls -la  # Should see version folders
   ```

## Manual Deployment (If Needed)

If you need to manually deploy a version:

```bash
# Install mike if not already installed
uv pip install mike

# Deploy version 1.3.1 as latest
mike deploy --push --update-aliases 1.3.1 latest

# Set default version
mike set-default --push latest

# List all versions
mike list
```

## Benefits

✅ **Versioned documentation**: Users can view docs for specific versions
✅ **Better user experience**: Version selector makes it easy to switch
✅ **Compatibility**: Supports both tag naming conventions
✅ **Automatic deployment**: No manual intervention needed
✅ **SEO friendly**: Each version has its own URL
✅ **Documentation history**: All versions remain accessible

## Next Steps

1. Commit the updated workflow file
2. Push to GitHub
3. Push the `py-cli-v1.3.1` tag
4. Verify the documentation deploys correctly with version selector
