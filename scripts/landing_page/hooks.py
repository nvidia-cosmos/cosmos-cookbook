"""MkDocs hooks for custom build steps."""
import json
import re
import shutil
from pathlib import Path
from html.parser import HTMLParser

# Allowed recipe tags (key:value). Defined early so parse_tags is always available.
ALLOWED_TAGS = frozenset({
    # General (not visible on site)
    "general:partner-recipe",
    "general:cookoff-recipe",
    "general:ai-friendly",
    # Domain (visible)
    "domain:robotics",
    "domain:autonomous-vehicles",
    "domain:smart-city",
    "domain:industrial",
    "domain:medical",
    "domain:fieldwork",
    "domain:cross-domain",
    # Technique (visible)
    "technique:data-augmentation",
    "technique:data-generation",
    "technique:prediction",
    "technique:reasoning",
    "technique:post-training",
    "technique:pre-training",
    "technique:data-curation-annotation",
    "technique:distillation",
    # Legacy tags (kept for backward compatibility with existing recipes)
    "technique:style-transfer",
    "technique:simulation",
    "technique:data-curation",
})


def parse_tags(tags_str):
    """Parse comma-separated key:value tags; return only allowed tags (lowercased for match)."""
    if tags_str is None:
        return []
    if not isinstance(tags_str, str):
        tags_str = str(tags_str)
    seen = set()
    result = []
    for raw in tags_str.split(","):
        tag = raw.strip().lower()
        if not tag:
            continue
        if tag in ALLOWED_TAGS and tag not in seen:
            result.append(tag)
            seen.add(tag)
    return result


class H1Parser(HTMLParser):
    """Extract the first H1 heading from HTML."""
    
    def __init__(self):
        super().__init__()
        self.in_h1 = False
        self.h1_text = []
        self.found_h1 = False
    
    def handle_starttag(self, tag, attrs):
        if tag == 'h1' and not self.found_h1:
            self.in_h1 = True
    
    def handle_endtag(self, tag):
        if tag == 'h1' and self.in_h1:
            self.in_h1 = False
            self.found_h1 = True
    
    def handle_data(self, data):
        if self.in_h1:
            self.h1_text.append(data)


class RecipeMetadataParser(HTMLParser):
    """Extract recipe metadata from HTML tables."""
    
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_thead = False
        self.in_tbody = False
        self.in_tr = False
        self.in_td = False
        self.in_th = False
        self.current_tag_class = None
        
        self.headers = []
        self.current_row = []
        self.rows = []
        self.current_cell_text = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'table':
            # Start tracking any table, we'll validate by headers later
            self.in_table = True
        elif self.in_table:
            if tag == 'thead':
                self.in_thead = True
            elif tag == 'tbody':
                self.in_tbody = True
            elif tag == 'tr':
                self.in_tr = True
                self.current_row = []
            elif tag == 'th' and self.in_thead:
                self.in_th = True
                self.current_cell_text = []
            elif tag == 'td' and self.in_tbody:
                self.in_td = True
                self.current_cell_text = []
    
    def handle_endtag(self, tag):
        if tag == 'table' and self.in_table:
            self.in_table = False
        elif tag == 'thead':
            self.in_thead = False
        elif tag == 'tbody':
            self.in_tbody = False
        elif tag == 'tr':
            if self.in_tr:
                if self.in_thead and self.current_row:
                    self.headers.extend(self.current_row)
                elif self.in_tbody and self.current_row:
                    self.rows.append(self.current_row)
                self.current_row = []
            self.in_tr = False
        elif tag == 'th':
            if self.in_th:
                self.current_row.append(''.join(self.current_cell_text).strip())
            self.in_th = False
            self.current_cell_text = []
        elif tag == 'td':
            if self.in_td:
                self.current_row.append(''.join(self.current_cell_text).strip())
            self.in_td = False
            self.current_cell_text = []
    
    def handle_data(self, data):
        if self.in_th or self.in_td:
            self.current_cell_text.append(data)


def extract_h1_title(html_content):
    """Extract the first H1 heading from HTML content."""
    parser = H1Parser()
    # Search first 100KB for H1 (should be early in the page)
    parser.feed(html_content[:100000])
    
    if parser.h1_text:
        return ''.join(parser.h1_text).strip()
    return None


def _content_fragment(html_content, max_chars=300000):
    """
    Extract the main page content fragment so we only pick images from the
    recipe/guide body, not from theme header/nav/logo. MkDocs Material uses
    .md-content__inner for the main content area.
    """
    m = re.search(
        r'<(?:article|div)[^>]*class="[^"]*md-content__inner[^"]*"[^>]*>',
        html_content,
        re.IGNORECASE,
    )
    if m:
        start = m.start()
        return html_content[start : start + max_chars]
    return html_content[:max_chars]


def _resolve_media_url(src, recipe_rel_url):
    """Resolve a relative media URL to site-root-relative (no leading slash)."""
    if not src or src.startswith(('data:', 'http://', 'https://')):
        return src if src else None
    recipe_dir = Path(recipe_rel_url).parent
    return (recipe_dir / src.lstrip("./")).as_posix()


def _get_src_from_tag(attrs_str):
    """Extract src or poster from an HTML tag's attribute string."""
    # (^|\s) so we match when src/poster is first attribute
    m = re.search(r'(^|\s)(?:src|poster)\s*=\s*["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)
    return m.group(2).strip() if m else None


def _has_featured_marker(attrs_str):
    """True if tag has media-featured=\"true\" or class containing media-featured."""
    # Match attribute name media-featured=, not text inside another attr value.
    # Preceded by start, space, or "> so we don't match e.g. alt="media-featured=..."
    if re.search(r'(^|[\s">])media-featured\s*=\s*["\']?true["\']?', attrs_str, re.IGNORECASE):
        return True
    # Fallback: class="media-featured"
    m = re.search(r'\sclass\s*=\s*["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)
    return bool(m and re.search(r'\bmedia-featured\b', m.group(1), re.IGNORECASE))


def extract_first_image_url(html_content, recipe_rel_url):
    """
    Thumbnail for Featured Recipes: prefer an image or video explicitly marked
    with media-featured="true"; otherwise use the first image in the main content.
    Returns a site-root-relative URL or None.
    """
    # 1) Look for media-featured in the ENTIRE document so we always find the
    #    marked image regardless of DOM structure (e.g. multiple md-content blocks).
    #    Use the LAST match so if multiple images are marked, we pick the intended one.
    featured_src = None
    for m in re.finditer(r'<img\s([^>]+)>', html_content, re.IGNORECASE):
        if _has_featured_marker(m.group(1)):
            src = _get_src_from_tag(m.group(1))
            if src:
                featured_src = src
    for m in re.finditer(r'<video\s([^>]+)>', html_content, re.IGNORECASE):
        if _has_featured_marker(m.group(1)):
            src = _get_src_from_tag(m.group(1))
            if src:
                featured_src = src
    if featured_src:
        return _resolve_media_url(featured_src, recipe_rel_url)
    # 2) Fallback: first <img> in the main content only (avoid theme/header images)
    fragment = _content_fragment(html_content)
    match = re.search(r'<img\s[^>]*?src=["\']([^"\']+)["\']', fragment, re.IGNORECASE)
    if not match:
        return None
    return _resolve_media_url(match.group(1).strip(), recipe_rel_url)


def extract_recipe_metadata(html_content, file_path):
    """Extract metadata from a recipe HTML file."""
    parser = RecipeMetadataParser()
    
    # Look for the first table in the content
    # Search first 50KB to catch tables after MkDocs template header
    parser.feed(html_content[:50000])
    
    if not parser.headers or not parser.rows:
        return None
    
    # Normalize headers
    normalized_headers = [h.lower().replace('**', '').strip() for h in parser.headers]
    
    # Check if this looks like a recipe metadata table
    # Should have Model, Workload, Use Case, and Category columns
    has_model = any('model' in h for h in normalized_headers)
    has_workload = any('workload' in h for h in normalized_headers)
    has_use_case = any('use case' in h or 'usecase' in h for h in normalized_headers)
    has_category = any('category' in h for h in normalized_headers)
    
    # Require Model, Workload, and Use Case (Category optional for backwards compatibility)
    if not (has_model and has_workload and has_use_case):
        return None
    
    # Convert to dict based on headers
    metadata = {}
    for row in parser.rows:
        if len(row) == len(parser.headers):
            for header, value in zip(normalized_headers, row):
                metadata[header] = value
    
    if not metadata:
        return None
    
    # Add the file path for linking
    metadata['url'] = str(file_path)
    
    return metadata


def _path_matches_word(path_str, word):
    """True if path_str contains word as a whole word (path segment or bounded by / - _)."""
    # Match word when preceded by start, /, - or _ and followed by end, /, - or _
    pattern = r'(^|[/\-_])' + re.escape(word) + r'($|[/\-_])'
    return bool(re.search(pattern, path_str))


def categorize_recipe(metadata, file_path):
    """Get recipe category from metadata, with fallback to auto-detection."""
    path_str = str(file_path).lower()
    
    # Try to get category from metadata first
    category = metadata.get('category', '').strip()
    
    # If no category in metadata, use auto-detection (backwards compatibility)
    if not category:
        category = "Vision AI"  # default
        if (
            _path_matches_word(path_str, 'robot')
            or _path_matches_word(path_str, 'warehouse')
            or _path_matches_word(path_str, 'gr00t')
            or _path_matches_word(path_str, 'manipulation')
        ):
            category = "Robotics"
        elif (
            _path_matches_word(path_str, 'av')
            or _path_matches_word(path_str, 'autonomous')
            or _path_matches_word(path_str, 'vehicle')
            or _path_matches_word(path_str, 'traffic')
            or _path_matches_word(path_str, 'its')
            or _path_matches_word(path_str, 'carla')
        ):
            category = "Autonomous Vehicles"
    
    # Get workload from metadata
    workload = metadata.get('workload', 'Inference').strip()
    
    return category, workload


def scan_recipes(site_dir):
    """Scan all recipe HTML files and extract metadata."""
    recipes = []
    site_path = Path(site_dir)
    
    # Scan recipes directory
    recipes_dir = site_path / "recipes"
    if not recipes_dir.exists():
        print("Warning: recipes directory not found")
        return recipes
    
    # Find all HTML files in recipes
    for html_file in recipes_dir.rglob("*.html"):
        # Skip SUMMARY.html and other non-recipe files
        if html_file.name in ['SUMMARY.html', 'index.html', 'all_recipes.html', 'additional_examples.html']:
            continue
        
        try:
            content = html_file.read_text(encoding='utf-8')
            metadata = extract_recipe_metadata(content, html_file.relative_to(site_path))
            
            if metadata:
                # Get relative URL from site root
                rel_url = str(html_file.relative_to(site_path))
                
                # Extract title from H1 heading
                title = extract_h1_title(content)
                if not title:
                    # Fallback: use use case from metadata
                    title = metadata.get('use case', '')
                    if not title:
                        # Final fallback: use model + workload
                        model = metadata.get('model', '')
                        workload = metadata.get('workload', '')
                        title = f"{model} - {workload}" if model and workload else html_file.stem.replace('_', ' ').title()
                
                # Get category and workload from metadata (or auto-detect)
                category, workload = categorize_recipe(metadata, html_file)
                
                # Thumbnail for Featured Recipes: first image in the recipe body (not stored in metadata table)
                thumbnail_url = extract_first_image_url(content, rel_url)

                # Build recipe entry (tags from optional **Tags** column, comma-separated)
                recipe = {
                    'name': title,
                    'type': 'Recipe',
                    'category': category,
                    'workload': workload,
                    'url': rel_url,
                    'model': metadata.get('model', ''),
                    'date': metadata.get('date', ''),
                    'tags': parse_tags(metadata.get('tags', '')),
                    'thumbnail': thumbnail_url or None,
                }
                
                recipes.append(recipe)
                print(f"  Found recipe: {title}")
        
        except Exception as e:
            print(f"  Warning: Could not parse {html_file}: {e}")
            continue

    # Append featured slot 6 (Prompt Guide) so it gets a thumbnail from its HTML
    slot6_path = site_path / "getting_started" / "prompt_guide" / "reason_guide.html"
    if slot6_path.exists():
        try:
            content = slot6_path.read_text(encoding="utf-8")
            rel_url = "getting_started/prompt_guide/reason_guide.html"
            title = extract_h1_title(content) or "Prompt Guide Cosmos Reason 2"
            thumbnail_url = extract_first_image_url(content, rel_url)
            recipes.append({
                "name": title,
                "type": "Recipe",
                "category": "Getting Started",
                "workload": "Prompt Guide",
                "url": rel_url,
                "model": "",
                "date": "",
                "tags": [],
                "thumbnail": thumbnail_url or None,
            })
            print(f"  Found featured slot 6: {title}")
        except Exception as e:
            print(f"  Warning: Could not parse featured slot 6: {e}")

    return recipes


def scan_section_pages(site_dir, section_path):
    """Scan all HTML pages in a section directory and return list of { title, url }."""
    site_path = Path(site_dir)
    section_dir = site_path / section_path
    if not section_dir.exists():
        return []
    pages = []
    for html_file in section_dir.rglob("*.html"):
        if html_file.name in ("SUMMARY.html", "index.html"):
            continue
        try:
            content = html_file.read_text(encoding="utf-8")
            title = extract_h1_title(content)
            if not title:
                title = html_file.stem.replace("_", " ").replace("-", " ").title()
            rel_url = str(html_file.relative_to(site_path))
            pages.append({"title": title, "url": rel_url})
        except Exception:
            continue
    return pages


def on_post_build(config):
    """Generate recipes.json and nav_pages.json after build. Always write nav_pages.json so the sidebar can load."""
    site_dir = config["site_dir"]
    nav_pages = {"getting_started": [], "recipes": [], "core_concepts": []}

    print("\n=== Custom Build Steps ===")

    try:
        # Scan recipes and generate recipes.json
        print("\n=== Scanning for recipes ===")
        recipes = scan_recipes(site_dir)

        if recipes:
            recipes_json_path = Path(site_dir) / "recipes.json"
            with open(recipes_json_path, 'w', encoding='utf-8') as f:
                json.dump(recipes, f, indent=2)
            print(f"\n✓ Generated recipes.json with {len(recipes)} recipes")
        else:
            print("\n⚠ No recipes found")

        # Build nav_pages for sidebar
        print("\n=== Building nav_pages.json ===")
        nav_pages["getting_started"] = scan_section_pages(site_dir, "getting_started")
        nav_pages["core_concepts"] = scan_section_pages(site_dir, "core_concepts")
        nav_pages["recipes"] = [
            {
                "title": r["name"],
                "url": r["url"],
                "category": r.get("category", "Vision AI"),
                "tags": r.get("tags") if isinstance(r.get("tags"), list) else [],
            }
            for r in recipes
        ]
    except Exception as e:
        print(f"\n⚠ Build step error (writing partial nav_pages.json): {e}")
        import traceback
        traceback.print_exc()

    nav_pages_path = Path(site_dir) / "nav_pages.json"
    with open(nav_pages_path, "w", encoding="utf-8") as f:
        json.dump(nav_pages, f, indent=2)
    print(f"✓ Generated nav_pages.json (getting_started: {len(nav_pages['getting_started'])}, recipes: {len(nav_pages['recipes'])}, core_concepts: {len(nav_pages['core_concepts'])})")

    # Inject nav data into every HTML page so the sidebar filter works without fetch (avoids path/CORS issues)
    nav_json = json.dumps(nav_pages).replace("</", "<\\/")
    inject_script = f'<script>window.__NAV_PAGES__={nav_json};</script>'
    site_path = Path(site_dir)
    injected = 0
    for html_file in site_path.rglob("*.html"):
        try:
            content = html_file.read_text(encoding="utf-8")
            if "</body>" in content and inject_script not in content:
                content = content.replace("</body>", inject_script + "\n</body>", 1)
                html_file.write_text(content, encoding="utf-8")
                injected += 1
        except Exception as e:
            print(f"  Warning: could not inject nav data into {html_file}: {e}")
    print(f"✓ Injected nav data into {injected} HTML page(s)")
    print("=== Build complete ===\n")
