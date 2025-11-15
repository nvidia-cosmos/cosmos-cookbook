# Contributing to Cosmos Cookbook

The Cosmos Cookbook is a community-driven resource for sharing practical knowledge about the NVIDIA Cosmos ecosystem. We welcome contributions including workflows, recipes, best practices, and domain-specific adaptations.

## What to Contribute

**Recipes** - Step-by-step guides for inference workflows or post-training (fine-tuning, LoRA, domain adaptation)

**Concepts** - Explanations of fundamental topics, techniques, architectural patterns, and tool documentation

**Improvements** - Bug fixes, documentation updates, broken links, or clarifications

## How to Contribute

- **Pull Request** - For complete contributions (use draft PRs for work in progress)
- **Issue** - For proposals, ideas, or reporting gaps in documentation

## Contribution Workflow

### 1. Fork and Set Up

Fork the [Cosmos Cookbook repository](https://github.com/nvidia-cosmos/cosmos-cookbook), then clone and configure:

```bash
git clone https://github.com/YOUR-USERNAME/cosmos-cookbook.git
cd cosmos-cookbook
git remote add upstream https://github.com/nvidia-cosmos/cosmos-cookbook.git

# Install dependencies (see README for details)
just install

# Verify setup
just serve-internal  # Visit http://localhost:8000
```

### 2. Create a Branch

```bash
git checkout -b recipe/descriptive-name  # or docs/, fix/, etc.
```

### 3. Make Changes

Add your content following the templates below, then test:

```bash
just serve-internal  # Preview changes
just test           # Run validation
```

### 4. Commit and Push

```bash
git add .
git commit -m "Add Transfer weather augmentation recipe"
git push origin recipe/descriptive-name
```

### 5. Create Pull Request

1. Visit your fork on GitHub and click **"Compare & pull request"**
2. Fill out the PR template with a clear title and description
3. Link related issues using `Closes #123` or `Relates to #456`
4. Submit the PR for review

### 6. Address Feedback

Update your branch based on review comments:

```bash
git add .
git commit -m "Address review feedback"
git push origin recipe/descriptive-name
```

The PR updates automatically. Once approved, maintainers will merge your contribution.

### Sync Your Fork

Before starting new work:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

## Content Templates and Organization

Use the appropriate template for your contribution:

- [Inference Recipe Template](assets/templates/inference_template.md) - Pre-trained model applications
- [Post-Training Recipe Template](assets/templates/post_training_template.md) - Fine-tuning and domain adaptation
- [Concept Template](assets/templates/concept_template.md) - Explanatory guides on fundamental topics

**Recipe structure:**

```
docs/recipes/{inference|post_training}/[model-name]/[recipe-name]/
├── {inference|post_training}.md  # Main content
├── setup.md                       # Optional setup guide
└── assets/                        # Media and configs
```

**Concept guides:** Place under `docs/core_concepts/[category]/` (e.g., `data_curation`, `post_training`, `evaluation`)

## Guidelines

**Dataset Licensing** - Verify proper licensing for any datasets used. Include clear attribution and licensing information.

**Code Review** - All submissions require review (typically within one week). Respond to feedback promptly and keep discussions professional.

## Developer Certificate of Origin (Optional)

You may optionally sign off your commits using `git commit -s`, which appends `Signed-off-by: Your Name <your@email.com>` to your commit message.

By signing off, you certify that you have the right to submit the contribution under the project's open source license, following the [Developer Certificate of Origin 1.1](https://developercertificate.org/).
