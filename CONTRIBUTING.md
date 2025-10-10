# How to Contribute

The Cosmos Cookbook is designed to create a dedicated space where the Cosmos team and community can openly share and contribute practical knowledge. We'd love to receive your patches and contributions to help build this valuable resource together.

## Open Source Contributions Welcome

We warmly welcome open source contributions to the Cosmos Cookbook! This is a community-driven cookbook documenting the Cosmos ecosystem, and we especially value your participation in the following ways:

### Sharing Your Success Stories and Practical Recipes

We'd love to hear your successful stories about how you've creatively used Cosmos models or repositories for different purposes, as well as practical recipes that you think others might benefit from learning. Whether you've:

- Adapted models for novel applications
- Discovered innovative workflows
- Achieved impressive results in specific domains
- Developed useful techniques or best practices
- Found creative solutions to common challenges

Your experiences and knowledge help make this cookbook more valuable for the entire community.

## Contribution Categories

### 1. **Recipes**

Step-by-step guides for specific tasks:

- **Inference**: Using pre-trained models for applications
- **Post-Training**: Domain adaptation via fine-tuning, LoRA, or other techniques

### 2. **Concepts**

Explanations of fundamental topics, techniques, and architectural patterns including workflows, best practices, and tool documentation.

### How to Contribute

- **Pull Request**: For complete contributions ready to merge (keep as draft until ready for review)
- **Issue with Proposal**: For ideas or incomplete stories you'd like help developing
- **Issue for Gaps**: For missing topics or knowledge gaps you'd like to see covered

We review all contributions within a week.

## Recipe Templates

Use the appropriate template when contributing a recipe:

- **📄 [Inference Recipe Template](assets/templates/inference_template.md)** - For pre-trained model applications
- **📄 [Post-Training Recipe Template](assets/templates/post_training_template.md)** - For domain adaptation workflows

### Recipe Organization

Recipes should be organized in the following directory structure:

```
docs/recipes/
├── inference/
│   └── [model-name]/
│       └── [recipe-name]/
│           ├── inference.md      # Main recipe
│           ├── setup.md          # Optional setup guide
│           └── assets/           # Images, videos, configs
└── post_training/
    └── [model-name]/
        └── [recipe-name]/
            ├── post_training.md  # Main recipe
            ├── setup.md          # Optional setup guide
            └── assets/           # Images, videos, configs
```

## Concept Template

- **📄 [Concept Template](assets/templates/concept_template.md)** - For explanatory guides on fundamental topics

Place concept guides in the appropriate subdirectory under `docs/core_concepts/` (e.g., `data_curation`, `post_training`, `evaluation`, `distillation`).

## Dataset Licensing

When contributing content with datasets:

1. Verify proper licensing for demonstration and promotional purposes
2. Include clear attribution and licensing information

By contributing, you agree to these dataset licensing terms.

## Testing

To serve the document locally, run

```shell
just serve-internal
```

To test your changes locally, run

```shell
just test
```

## Code Reviews

All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

## Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
