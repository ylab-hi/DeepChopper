# Documentation Versioning with Mike

DeepChopper uses [mike](https://github.com/jimporter/mike) to manage versioned documentation.

## Quick Start

### Deploy Development Version

```bash
# Deploy current docs as 'dev'
uv run mike deploy dev

# Deploy and set as default
uv run mike deploy dev --push --update-aliases
```

### Deploy Release Version

```bash
# Deploy version 1.2.9 and mark as 'latest'
uv run mike deploy 1.2.9 latest --push --update-aliases

# Set default version
uv run mike set-default stable --push
```

### List All Versions

```bash
uv run mike list
```

### Delete a Version

```bash
uv run mike delete old-version --push
```

## Version Strategy

### Version Types

1. **Stable** - Latest stable release (recommended for users)
2. **Latest** - Alias for most recent release
3. **Dev** - Development branch (for contributors)
4. **X.Y.Z** - Specific version numbers

### Naming Convention

- **Major releases**: `1.0`, `2.0`, etc.
- **Minor releases**: `1.1`, `1.2`, etc.
- **Patch releases**: `1.2.1`, `1.2.2`, etc.
- **Development**: `dev`

## Automated Deployment

Documentation is automatically versioned via GitHub Actions:

### On Tag Push

```yaml
# When you push a tag like v1.2.9
git tag v1.2.9
git push origin v1.2.9

# GitHub Actions will:
# 1. Deploy docs as version 1.2.9
# 2. Update 'latest' alias
# 3. Set 'stable' as default
```

### On Main Branch Push

```yaml
# When you push to main
git push origin main

# GitHub Actions will:
# 1. Deploy docs as 'dev'
# 2. Keep existing versions
```

## Manual Deployment

### First Time Setup

```bash
# Initialize mike (creates gh-pages branch)
uv run mike deploy --push dev
```

### Deploy New Version

```bash
# 1. Update version in pyproject.toml
# 2. Deploy with mike
uv run mike deploy 1.3.0 latest --push --update-aliases

# 3. Optionally set as default
uv run mike set-default 1.3.0 --push
```

### Update Existing Version

```bash
# Update dev docs
uv run mike deploy dev --push --update-aliases
```

## Viewing Versions Locally

### Serve All Versions

```bash
# Serve versioned docs
uv run mike serve

# Access at http://localhost:8000/
```

### Serve Specific Version

```bash
# Build and serve specific version
uv run mkdocs serve
```

## Version Selector

Users can select versions via the dropdown in the header:

- Click the version selector
- Choose desired version
- Documentation updates automatically

## Configuration

### mkdocs.yml

```yaml
extra:
  version:
    provider: mike
    default: stable
```

This enables:
- Version dropdown in header
- Automatic version detection
- Default version redirect

## Best Practices

### Do's

✅ Deploy stable versions with release tags
✅ Keep 'dev' updated with main branch
✅ Use semantic versioning (X.Y.Z)
✅ Document breaking changes
✅ Test before deploying

### Don'ts

❌ Don't delete versions users depend on
❌ Don't skip version numbers
❌ Don't deploy without testing
❌ Don't overwrite existing versions without reason

## Troubleshooting

### Version Not Showing

```bash
# Check deployed versions
uv run mike list

# Verify gh-pages branch
git branch -r | grep gh-pages
```

### Deploy Failed

```bash
# Check git configuration
git config user.name
git config user.email

# Verify gh-pages branch exists
git fetch origin gh-pages:gh-pages
```

### Version Selector Missing

1. Check `mkdocs.yml` has `version.provider: mike`
2. Verify deployed with `mike deploy`
3. Clear browser cache

## Examples

### Release Workflow

```bash
# 1. Finish development
git checkout main
git pull origin main

# 2. Update version
# Edit pyproject.toml: version = "1.3.0"

# 3. Commit and tag
git add pyproject.toml
git commit -m "Bump version to 1.3.0"
git tag v1.3.0

# 4. Push (triggers auto-deployment)
git push origin main --tags
```

### Hotfix Workflow

```bash
# 1. Fix issue on release branch
git checkout -b hotfix/1.2.10

# 2. Deploy hotfix docs
uv run mike deploy 1.2.10 --push

# 3. Update latest alias if needed
uv run mike alias 1.2.10 latest --push
```

## Resources

- [Mike Documentation](https://github.com/jimporter/mike)
- [MkDocs Versioning Guide](https://www.mkdocs.org/user-guide/deploying-your-docs/)
- [Semantic Versioning](https://semver.org/)

## Support

For issues with versioning:
1. Check [GitHub Actions logs](.github/workflows/docs-versions.yml)
2. Verify `gh-pages` branch
3. Test locally with `mike serve`
4. Open an issue if problem persists
