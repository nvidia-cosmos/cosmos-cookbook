# Recipe Tags

Tags are used to categorize and filter recipes, allowing for user navigation in both the Cookbook Docs sidebar and landing page. 

## Allowed Tags

- **General:**
    - `general:partner-recipe`: Recipe contributed by a partner organization (will be a candidate for the Featured Recipes section on the landing page)
    - `general:cookoff-recipe`: Recipe contributed to a Cosmos Cookoff event (will be a candidate for the Featured Recipes section on the landing page)
    - `general:ai-friendly`: Recipe is AI-friendly and can be used by AI agents
- **Domain:**
    - `domain:robotics`: Recipe is related to robotics
    - `domain:autonomous-vehicles`: Recipe is related to autonomous vehicles
    - `domain:smart-city`: Recipe is related to smart city
    - `domain:industrial`: Recipe is related to industrial
    - `domain:medical`: Recipe is related to medical
    - `domain:fieldwork`: Recipe is related to fieldwork
    - `domain:cross-domain`: Recipe is related to cross-domain
- **Technique:**
    - `technique:data-augmentation`: Recipe is related to data augmentation
    - `technique:data-generation`: Recipe is related to data generation
    - `technique:prediction`: Recipe is related to prediction
    - `technique:reasoning`: Recipe is related to reasoning
    - `technique:post-training`: Recipe is related to post-training
    - `technique:pre-training`: Recipe is related to pre-training
    - `technique:data-curation-annotation`: Recipe is related to data curation annotation
    - `technique:distillation`: Recipe is related to distillation

You should add one "domain" and one "technique" tag to your recipe. The "general" tag is optional.

## Example

Here's an example of how to use tags in a recipe.

```markdown
| **Model** | **Workload** | **Use Case** | **Tags** |
|-----------|--------------|--------------|----------|
| [Cosmos Reason 2 8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)| Inference | Large scale video search and summarization. | domain:smart-city, technique:reasoning, general:cookoff-recipe |
```

## Media Tagging for Featured Recipes

For Featured Recipes (recipes with `general:partner-recipe` or `general:cookoff-recipe`), the landing page uses a thumbnail image (or video poster) from the recipe body. You can control which image or video is used as the thumbnail using the following markers:

- **Explicit marker:** Add `media-featured="true"` and/or `class="media-featured"` to the HTML of the image or video you want to use, as shown in the following example:  
   
  ```html 
  `<img src="assets/hero.png" alt="Overview" class="media-featured" media-featured="true">`
  ```

- **Default behavior:** If no element is marked, the **first image** in the recipe is used. If there is no image, a placeholder with the recipe initial is shown.