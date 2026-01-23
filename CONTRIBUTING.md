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

### Directory Structure

The Cosmos Cookbook is organized into three main content areas:

#### 1. **Getting Started** (`docs/getting_started/`)

Enablement documents that help users set up and start using Cosmos models quickly.

**Use for:**

- Installation and setup guides
- Quick start tutorials
- Platform-specific deployment guides (e.g., Brev, cloud platforms)
- Prerequisites and environment setup

#### 2. **Core Concepts** (`docs/core_concepts/`)

Didactic knowledge-sharing content that explains fundamental topics, techniques, and architectural patterns.

**Use for:**

- Explanations of key concepts and techniques
- Architectural deep-dives
- Best practices and guidelines
- Technical reference documentation

**Structure:**

```
docs/core_concepts/
├── [category]/                 # e.g., data_curation, post_training, evaluation
│   ├── overview.md            # Category overview
│   ├── [topic].md             # Individual concept guides
│   └── assets/                # Supporting media
```

**Example categories:** `data_curation`, `post_training`, `control_modalities`, `evaluation`, `distillation`

#### 3. **Recipes** (`docs/recipes/`)

Step-by-step practical guides demonstrating real-world applications and workflows.

**Use for:**

- Inference workflows using pre-trained models
- Post-training/fine-tuning guides
- End-to-end workflows
- Data curation pipelines

**Structure:**

```
docs/recipes/
├── inference/                  # Inference workflows
│   └── [model-name]/          # e.g., predict2, transfer2_5, reason1
│       └── [recipe-name]/
│           ├── inference.md   # Main content
│           ├── setup.md       # Optional setup guide
│           └── assets/        # Media and configs
├── post_training/             # Training/fine-tuning workflows
│   └── [model-name]/
│       └── [recipe-name]/
│           ├── post_training.md
│           ├── setup.md
│           └── assets/
├── data_curation/             # Data processing pipelines
│   └── [recipe-name]/
│       ├── data_curation.md
│       └── assets/
└── end2end/                   # Complete workflows
    └── [workflow-name]/
        ├── workflow_e2e.md
        └── assets/
```

### Content Templates

Use the appropriate template for your contribution:

- [Inference Recipe Template](assets/templates/inference_template.md) - Pre-trained model applications
- [Post-Training Recipe Template](assets/templates/post_training_template.md) - Fine-tuning and domain adaptation
- [Concept Template](assets/templates/concept_template.md) - Explanatory guides on fundamental topics

### Updating Index Files (Optional)

When adding new recipes or documentation pages, you may optionally update the following index files to make your content more discoverable:

#### Files to Update

1. **`docs/index.md`** - The main landing page
   - Add your recipe to the appropriate table in the "Case Study Recipes" section
   - Update the "Latest Updates" table if your contribution is recent

2. **`README.md`** - Repository README
   - Add an entry to the "Latest Updates" table
   - Ensure the description matches the entry in `docs/index.md`

3. **`docs/recipes/all_recipes.md`** - Recipe gallery page
   - Add a new recipe card in the appropriate category section (Robotics, Autonomous Vehicles, or Vision AI)
   - Include the recipe title, link, media (image/video), and tag (Inference, Post-Training, Curation, or Workflow)

#### Example Update

For a new recipe at `docs/recipes/inference/transfer2_5/my-new-recipe/inference.md`:

```markdown
<!-- In docs/index.md under appropriate model section -->
| **Inference** | Description of my new recipe | [My New Recipe](recipes/inference/transfer2_5/my-new-recipe/inference.md) |

<!-- In docs/recipes/all_recipes.md under appropriate category -->
<a class="recipe-card" href="./inference/transfer2_5/my-new-recipe/inference.md">
  <div class="recipe-media recipe-media--image" aria-hidden="true">
    <img src="./inference/transfer2_5/my-new-recipe/assets/hero.png" alt="" loading="lazy" />
  </div>
  <div class="recipe-title">My New Recipe Title</div>
  <div class="recipe-tag recipe-tag--inference">Inference</div>
</a>
```

**Note:** This step is optional. If you don't update these index files, repository maintainers will add your contribution to the appropriate locations in follow-up modifications.

## Guidelines

**Dataset Licensing** - Verify proper licensing for any datasets used. Include clear attribution and licensing information.

**Code Review** - All submissions require review (typically within one week). Respond to feedback promptly and keep discussions professional.

## Developer Certificate of Origin (Optional)

You may optionally sign off your commits using `git commit -s`, which appends `Signed-off-by: Your Name <your@email.com>` to your commit message.

By signing off, you certify that you have the right to submit the contribution under the project's open source license, following the [Developer Certificate of Origin 1.1](https://developercertificate.org/).
