# Cosmos Cookbook Repository Setup Guide

This guide covers the complete setup for the **Cosmos Cookbook** repository, including CI/CD pipelines, pre-commit hooks, and documentation deployment.

## 🚀 Repository Features

### ✅ Completed Setup

- **Pre-commit hooks** for code quality and consistency
- **GitHub Actions** for automated testing and deployment
- **Documentation deployment** via GitHub Pages
- **Issue and PR templates** for structured contributions
- **Automated code formatting** (Black, isort)
- **Markdown linting** for documentation quality

## 📋 Setup Checklist

### 1. GitHub Repository Settings

#### Enable GitHub Pages

1. Go to **Settings** → **Pages**
2. Set **Source** to "GitHub Actions"
3. The documentation will be automatically deployed to: `https://nvidia-cosmos.github.io/cosmos-cookbook/`

#### Branch Protection Rules (Recommended)

1. Go to **Settings** → **Branches**
2. Add rule for `main` branch:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

### 2. Local Development Setup

#### Prerequisites

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install pre-commit
uv tool install pre-commit
```

#### Repository Setup

```bash
# Clone the repository
git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git
cd cosmos-cookbook

# Install pre-commit hooks
pre-commit install

# Test the setup
pre-commit run --all-files
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

#### 1. PR Quality Checks (`.github/workflows/pr-checks.yml`)

**Triggers:** Pull requests to `main`, pushes to `main`

**Jobs:**

- **Pre-commit checks**: Runs all linting and formatting tools
- **Documentation build test**: Ensures docs build successfully

#### 2. Documentation Deployment (`.github/workflows/deploy-docs.yml`)

**Triggers:** Pushes to `main`, manual dispatch

**Jobs:**

- **Build**: Generates documentation with MkDocs
- **Deploy**: Publishes to GitHub Pages

### Pre-commit Hooks (`.pre-commit-config.yaml`)

**Automated Quality Checks:**

- **Trailing whitespace removal**
- **End-of-file fixing**
- **YAML validation**
- **Large file detection**
- **Merge conflict detection**
- **Markdown linting** (markdownlint)
- **YAML formatting** (yamlfmt)
- **Python formatting** (Black, isort)
- **Python linting** (flake8)

## 📝 Contributing Workflow

### For Contributors

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Test** locally: `pre-commit run --all-files`
5. **Commit** with descriptive messages
6. **Push** to your fork: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

### For Maintainers

1. **Review** PR using the provided template
2. **Check** that all CI checks pass
3. **Merge** when ready (squash merge recommended)
4. **Documentation** automatically deploys on merge to `main`

## 🛠️ Development Commands

### Pre-commit Management

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run markdownlint --all-files
pre-commit run black --all-files

# Update hook versions
pre-commit autoupdate

# Skip hooks for emergency commits (not recommended)
git commit -m "emergency fix" --no-verify
```

### Documentation Development

```bash
# Install documentation dependencies
uv venv
source .venv/bin/activate
uv pip install mkdocs mkdocs-material mkdocs-mermaid2-plugin

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

## 📊 Quality Standards

### Markdown Guidelines

- Use proper headings (`#`, `##`, `###`) instead of bold text
- Include language specifiers in code blocks: ````python`,````bash`
- Keep line lengths reasonable (no strict limit)
- Include alt text for images when possible

### Python Code Standards

- **Formatting**: Black (88 character line length)
- **Import sorting**: isort (compatible with Black)
- **Linting**: flake8 (relaxed for existing code)
- **Type hints**: Encouraged for new code

### Commit Message Format

```
type(scope): description

- Bullet point for changes
- Another change
- More details if needed
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## 🔧 Troubleshooting

### Pre-commit Issues

```bash
# Clear pre-commit cache
pre-commit clean

# Reinstall hooks
pre-commit uninstall
pre-commit install

# Skip problematic hooks temporarily
SKIP=flake8 git commit -m "temporary skip"
```

### GitHub Actions Failures

1. Check the **Actions** tab for detailed logs
2. Common issues:
   - **Documentation build**: Missing dependencies in `mkdocs.yml`
   - **Pre-commit**: New linting errors in changed files
   - **Permissions**: GitHub Pages deployment permissions

### Documentation Deployment

- **First deployment**: May take 5-10 minutes to appear
- **Updates**: Usually reflect within 2-3 minutes
- **Custom domain**: Configure in repository settings if needed

## 📚 Additional Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [Markdownlint Rules](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)

## 🎯 Next Steps

1. **Test the pipeline** by creating a test PR
2. **Customize** the pre-commit hooks for your specific needs
3. **Add** additional quality checks as needed
4. **Train** team members on the new workflow
5. **Monitor** the GitHub Actions usage and optimize as needed

---

**Repository Status**: ✅ Fully configured with CI/CD pipeline
**Documentation**: 🌐 Available at <https://nvidia-cosmos.github.io/cosmos-cookbook/>
**Last Updated**: October 1, 2025
