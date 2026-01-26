"""MkDocs hooks for custom build steps."""
import json
import re
import shutil
from pathlib import Path
from html.parser import HTMLParser


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


def categorize_recipe(metadata, file_path):
    """Get recipe category from metadata, with fallback to auto-detection."""
    path_str = str(file_path).lower()
    
    # Try to get category from metadata first
    category = metadata.get('category', '').strip()
    
    # If no category in metadata, use auto-detection (backwards compatibility)
    if not category:
        category = "Vision AI"  # default
        if 'robot' in path_str or 'warehouse' in path_str or 'gr00t' in path_str or 'manipulation' in path_str:
            category = "Robotics"
        elif 'av' in path_str or 'autonomous' in path_str or 'vehicle' in path_str or 'traffic' in path_str or 'its' in path_str or 'carla' in path_str:
            category = "Autonomous Vehicles"
        
        # Check use case field too
        use_case = metadata.get('use case', '').lower()
        if 'robot' in use_case or 'manipulation' in use_case or 'warehouse' in use_case:
            category = "Robotics"
        elif 'autonomous' in use_case or 'vehicle' in use_case or 'traffic' in use_case or 'av' in use_case:
            category = "Autonomous Vehicles"
    
    # Get workload directly from metadata
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
        if html_file.name in ['SUMMARY.html', 'index.html', 'all_recipes.html']:
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
                
                # Build recipe entry
                recipe = {
                    'name': title,
                    'type': 'Recipe',
                    'category': category,
                    'workload': workload,
                    'url': rel_url,
                    'model': metadata.get('model', ''),
                    'date': metadata.get('date', '')  # Optional field
                }
                
                recipes.append(recipe)
                print(f"  Found recipe: {title}")
        
        except Exception as e:
            print(f"  Warning: Could not parse {html_file}: {e}")
            continue
    
    return recipes


def on_post_build(config):
    """Copy custom HTML files and generate recipes.json after build."""
    site_dir = config["site_dir"]
    
    print("\n=== Custom Build Steps ===")
    
    # 1. Copy index.html without MkDocs template to override the default
    source = Path("docs/index.html")
    dest = Path(site_dir) / "index.html"
    
    if source.exists():
        shutil.copy2(source, dest)
        print(f"✓ Copied {source} to {dest}")
    
    # 2. Scan recipes and generate recipes.json
    print("\n=== Scanning for recipes ===")
    recipes = scan_recipes(site_dir)
    
    if recipes:
        recipes_json_path = Path(site_dir) / "recipes.json"
        with open(recipes_json_path, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, indent=2)
        print(f"\n✓ Generated recipes.json with {len(recipes)} recipes")
    else:
        print("\n⚠ No recipes found")
    
    print("=== Build complete ===\n")
