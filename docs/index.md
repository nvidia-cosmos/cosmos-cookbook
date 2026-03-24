<!-- markdownlint-disable MD037 MD038 --><!-- Embedded CSS `* {` triggers MD037; JS template literals in <script> trigger MD038 / crash markdownlint-cli -->

  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: var(--md-text-font, "NVIDIA Sans", -apple-system, sans-serif);
      background-color: var(--md-default-bg-color, #ffffff);
      color: var(--md-default-fg-color, #333333);
      line-height: 1.6;
    }

    /* Extend content area to the right for more horizontal space */
    .landing-page-wrapper {
      --landing-right-bleed: clamp(2rem, 12vw, 18rem);
      width: calc(100% + var(--landing-right-bleed));
      margin-right: calc(-1 * var(--landing-right-bleed));
      padding-right: 0.5rem;
    }

    /* Header Section */
    .hero {
      background: linear-gradient(135deg, #76b900 0%, #5f9300 100%);
      color: white;
      text-align: center;
      padding: 1rem 0.75rem;
    }

    .hero h1 {
      font-size: 1.4rem;
      font-weight: 700;
      margin-bottom: 0.2rem;
    }

    .hero p {
      font-size: 0.9rem;
      font-weight: 400;
      opacity: 0.95;
    }

    /* Container */
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 0.75rem;
    }

    /* Section Headers */
    .section-header {
      background: linear-gradient(135deg, #76b900 0%, #5f9300 100%);
      color: white;
      padding: 0.5rem 0.75rem;
      margin: 1rem 0 0.75rem 0;
      font-size: 1.1rem;
      font-weight: 600;
      text-align: center;
    }

    /* Featured Recipes: two rows of three */
    .featured-recipes {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.75rem;
      margin-bottom: 1.5rem;
    }

    .recipe-card {
      border: 2px solid #e0e0e0;
      padding: 0;
      transition: all 0.3s ease;
      cursor: pointer;
      text-decoration: none;
      color: inherit;
      display: block;
      overflow: hidden;
    }

    .recipe-card:hover {
      border-color: #76b900;
      box-shadow: 0 4px 12px rgba(118, 185, 0, 0.2);
      transform: translateY(-2px);
    }

    .recipe-thumbnail {
      width: 100%;
      height: 120px;
      min-height: 120px;
      max-height: 120px;
      object-fit: cover;
      object-position: center;
      background: #f5f5f5;
      display: block;
      flex-shrink: 0;
    }

    .recipe-content {
      padding: 0.6rem;
    }

    .recipe-title {
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 0.3rem;
      color: #76b900;
      min-height: 2.2rem;
      line-height: 1.3;
    }

    .recipe-description {
      font-size: 0.8rem;
      color: var(--md-default-fg-color--light, #666666);
      line-height: 1.3;
    }

    /* All Recipes Section */
    .all-recipes-section {
      margin-top: 1.5rem;
    }

    .docs-link-container {
      text-align: center;
      margin: 1.5rem 0 1rem 0;
    }

    .docs-link-btn {
      display: inline-block;
      padding: 0.5rem 1.2rem;
      background: white;
      border: 2px solid #76b900;
      color: #76b900;
      font-size: 0.85rem;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.3s ease;
      font-family: inherit;
      cursor: pointer;
    }

    .docs-link-btn:hover {
      background: #76b900;
      color: white;
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(118, 185, 0, 0.3);
    }

    .filter-controls {
      display: flex;
      gap: 0.5rem;
      justify-content: center;
      flex-wrap: wrap;
      margin-bottom: 1rem;
      position: relative;
      z-index: 2;
    }

    .filter-btn {
      padding: 0.4rem 1rem;
      background: white;
      border: 2px solid #76b900;
      color: #76b900;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
      font-family: inherit;
      pointer-events: auto;
      position: relative;
    }

    .filter-btn:hover,
    .filter-btn.active {
      background: #76b900;
      color: white;
    }

    /* Search Bar */
    .search-container {
      margin-bottom: 2rem;
      max-width: 600px;
      margin-left: auto;
      margin-right: auto;
    }

    .search-input {
      width: 100%;
      padding: 0.4rem 0.6rem;
      font-size: 0.85rem;
      border: 2px solid #e0e0e0;
      font-family: inherit;
      transition: border-color 0.3s ease;
    }

    .search-input:focus {
      outline: none;
      border-color: #76b900;
    }

    .search-input::placeholder {
      color: #999999;
    }

    /* Pagination Controls */
    .pagination-controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
      padding: 0 0.25rem;
    }

    .pagination-info {
      font-size: 0.8rem;
      color: var(--md-default-fg-color--light, #666666);
    }

    .pagination-buttons {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }

    .page-select {
      padding: 0.5rem;
      border: 1px solid #e0e0e0;
      font-family: inherit;
    }

    .pagination-btn {
      padding: 0.5rem 0.75rem;
      background: white;
      border: 1px solid #e0e0e0;
      cursor: pointer;
      transition: all 0.3s ease;
    }

    .pagination-btn:hover:not(:disabled) {
      background: #76b900;
      color: white;
      border-color: #76b900;
    }

    .pagination-btn:disabled {
      opacity: 0.3;
      cursor: not-allowed;
    }

    /* Recipe Table */
    .recipe-table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 1.5rem;
      background: white;
      font-size: 0.85rem;
    }

    .recipe-table thead {
      background: #f5f5f5;
    }

    .recipe-table th {
      text-align: left;
      padding: 0.5rem 0.6rem;
      font-weight: 600;
      border-bottom: 2px solid #e0e0e0;
      font-size: 0.85rem;
    }

    .recipe-table td {
      padding: 0.5rem 0.6rem;
      border-bottom: 1px solid #e0e0e0;
    }

    .recipe-table tbody tr {
      transition: background-color 0.2s ease;
    }

    .recipe-table tbody tr:hover {
      background: #f9f9f9;
    }

    .recipe-table a {
      color: #76b900;
      text-decoration: none;
      font-weight: 600;
    }

    .recipe-table a:hover {
      text-decoration: underline;
    }

    .recipe-tag {
      display: inline-block;
      padding: 0.15rem 0.5rem;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 3px;
      background: #e0e0e0;
      color: #333;
    }

    .recipe-tag--inference {
      background: #e3f2fd;
      color: #1976d2;
    }

    .recipe-tag--post-training {
      background: #f3e5f5;
      color: #7b1fa2;
    }

    .recipe-tag--curation {
      background: #fff3e0;
      color: #f57c00;
    }

    .recipe-tag--workload {
      background: #e8f5e9;
      color: #388e3c;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .hero h1 {
        font-size: 1.2rem;
      }

      .hero p {
        font-size: 0.85rem;
      }

      .hero {
        padding: 0.75rem 0.5rem;
      }

      .featured-recipes {
        grid-template-columns: 1fr;
      }

      .filter-controls {
        flex-direction: column;
      }

      .filter-btn {
        width: 100%;
      }

      .recipe-table {
        font-size: 0.75rem;
      }

      .recipe-table th,
      .recipe-table td {
        padding: 0.4rem;
      }
    }

    /* Dark mode support */
    [data-md-color-scheme="dark"] body {
      background-color: var(--md-default-bg-color, #1a1a1a);
      color: var(--md-default-fg-color, #ffffff);
    }

    [data-md-color-scheme="dark"] .recipe-card {
      border-color: #333;
      background: #1a1a1a;
    }

    [data-md-color-scheme="dark"] .recipe-table {
      background: #1a1a1a;
    }

    [data-md-color-scheme="dark"] .recipe-table thead {
      background: #2a2a2a;
    }

    [data-md-color-scheme="dark"] .recipe-table tbody tr:hover {
      background: #2a2a2a;
    }

    [data-md-color-scheme="dark"] .docs-link-btn {
      background: #1a1a1a;
      border-color: #76b900;
      color: #76b900;
    }

    [data-md-color-scheme="dark"] .docs-link-btn:hover {
      background: #76b900;
      color: white;
    }

    /* Dark mode: All Recipes search bar */
    [data-md-color-scheme="dark"] .search-input {
      background: #1a1a1a;
      color: #e8e8e8;
      border-color: #444;
    }

    [data-md-color-scheme="dark"] .search-input::placeholder {
      color: #a0a0a0;
    }

    [data-md-color-scheme="dark"] .search-input:focus {
      border-color: #76b900;
    }
  </style>

  <div class="landing-page-wrapper">
  <!-- Hero Section -->
  <div class="hero">
    <h1>NVIDIA Cosmos Cookbook</h1>
    <p>A practical guide with recipes to build, fine-tune, and deploy physical-AI</p>
  </div>

  <!-- Featured Recipes Section (loaded dynamically: partner/cookoff tags, 6 most recent by date) -->
  <div class="container">
    <div class="section-header">Featured Recipes</div>

    <div class="featured-recipes" id="featuredRecipesContainer">
      <!-- Populated by JavaScript from recipes.json -->
      <div class="featured-recipes-loading" style="grid-column: 1 / -1; text-align: center; padding: 2rem; color: #999;">Loading featured recipes…</div>
    </div>

    <!-- All Recipes Section -->
    <div class="all-recipes-section">
      <div class="section-header">All Recipes</div>

      <div class="filter-controls">
        <button class="filter-btn active" data-domain="all">All</button>
        <button class="filter-btn" data-domain="domain:robotics">Robotics</button>
        <button class="filter-btn" data-domain="domain:autonomous-vehicles">Autonomous Vehicles</button>
        <button class="filter-btn" data-domain="domain:smart-city">Smart City</button>
        <button class="filter-btn" data-domain="domain:industrial">Industrial</button>
        <button class="filter-btn" data-domain="domain:medical">Medical</button>
      </div>

      <div class="search-container">
        <input type="text" class="search-input" placeholder="🔍 Search for recipes, concepts, and prompts" id="searchInput">
      </div>

      <div class="pagination-controls">
        <div class="pagination-info">
          <span id="paginationInfo">1 - 15 of 282 items</span>
        </div>
        <div class="pagination-buttons">
          <button class="pagination-btn" id="firstPage">|&lt;</button>
          <button class="pagination-btn" id="prevPage">&lt;</button>
          <select class="page-select" id="pageSelect">
            <option value="1">1</option>
          </select>
          <button class="pagination-btn" id="nextPage">&gt;</button>
          <button class="pagination-btn" id="lastPage">&gt;|</button>
        </div>
      </div>

      <table class="recipe-table">
        <thead>
          <tr>
            <th>Recipe Name</th>
            <th>Workload</th>
            <th>Category</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody id="recipeTableBody">
          <!-- Recipes will be populated by JavaScript -->
        </tbody>
      </table>
    </div>
  </div>

  <script>
    // Recipe data - loaded dynamically from recipes.json
    let recipes = [];

    // State
    let currentPage = 1;
    let itemsPerPage = 15;
    let currentDomain = 'all';
    let searchQuery = '';
    let isLoading = true;

    // Escape for safe insertion into HTML (prevents XSS when using innerHTML)
    function escapeHtml(str) {
      if (str == null || typeof str !== 'string') return '';
      return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    // Get filtered recipes (by Domain tag, same as sidebar; recipes with no tags appear under all domains)
    function getFilteredRecipes() {
      return recipes.filter(recipe => {
        const tags = recipe.tags && Array.isArray(recipe.tags) ? recipe.tags : [];
        const matchesDomain = currentDomain === 'all' || tags.length === 0 || tags.indexOf(currentDomain) !== -1;
        const matchesSearch = searchQuery === '' ||
          recipe.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          (recipe.type && recipe.type.toLowerCase().includes(searchQuery.toLowerCase())) ||
          (recipe.category && recipe.category.toLowerCase().includes(searchQuery.toLowerCase()));
        return matchesDomain && matchesSearch;
      });
    }

    // Render recipe table
    function renderRecipeTable() {
      const tbody = document.getElementById('recipeTableBody');

      // Show loading state
      if (isLoading) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; padding: 2rem; color: #999;">Loading recipes...</td></tr>';
        document.getElementById('paginationInfo').textContent = 'Loading...';
        return;
      }

      const filteredRecipes = getFilteredRecipes();
      const totalItems = filteredRecipes.length;
      const totalPages = Math.ceil(totalItems / itemsPerPage);

      // Adjust current page if necessary
      if (currentPage > totalPages) {
        currentPage = Math.max(1, totalPages);
      }

      const startIdx = (currentPage - 1) * itemsPerPage;
      const endIdx = Math.min(startIdx + itemsPerPage, totalItems);
      const pageRecipes = filteredRecipes.slice(startIdx, endIdx);

      tbody.innerHTML = '';

      pageRecipes.forEach(recipe => {
        const row = document.createElement('tr');
        const url = escapeHtml(recipe.url);
        const name = escapeHtml(recipe.name);
        const workload = escapeHtml(recipe.workload);
        const category = escapeHtml(recipe.category);
        const date = escapeHtml(recipe.date);
        row.innerHTML = `
          <td><a href="${url}">${name}</a></td>
          <td>${workload}</td>
          <td>${category}</td>
          <td>${date}</td>
        `;
        tbody.appendChild(row);
      });

      // Update pagination info
      document.getElementById('paginationInfo').textContent =
        `${startIdx + 1} - ${endIdx} of ${totalItems} items`;

      // Update page select
      const pageSelect = document.getElementById('pageSelect');
      pageSelect.innerHTML = '';
      for (let i = 1; i <= totalPages; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = i;
        if (i === currentPage) option.selected = true;
        pageSelect.appendChild(option);
      }

      // Update button states
      document.getElementById('firstPage').disabled = currentPage === 1;
      document.getElementById('prevPage').disabled = currentPage === 1;
      document.getElementById('nextPage').disabled = currentPage === totalPages || totalPages === 0;
      document.getElementById('lastPage').disabled = currentPage === totalPages || totalPages === 0;
    }

    // Domain filter buttons (match sidebar Domain options) — use delegation so every button stays clickable
    const filterControls = document.querySelector('.all-recipes-section .filter-controls');
    if (filterControls) {
      filterControls.addEventListener('click', (e) => {
        const btn = e.target.closest('.filter-btn');
        if (!btn) return;
        document.querySelectorAll('.all-recipes-section .filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentDomain = btn.getAttribute('data-domain') || 'all';
        currentPage = 1;
        renderRecipeTable();
      });
    }

    // Search input
    document.getElementById('searchInput').addEventListener('input', (e) => {
      searchQuery = e.target.value;
      currentPage = 1;
      renderRecipeTable();
    });

    // Pagination buttons
    document.getElementById('firstPage').addEventListener('click', () => {
      currentPage = 1;
      renderRecipeTable();
    });

    document.getElementById('prevPage').addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        renderRecipeTable();
      }
    });

    document.getElementById('nextPage').addEventListener('click', () => {
      const totalPages = Math.ceil(getFilteredRecipes().length / itemsPerPage);
      if (currentPage < totalPages) {
        currentPage++;
        renderRecipeTable();
      }
    });

    document.getElementById('lastPage').addEventListener('click', () => {
      const totalPages = Math.ceil(getFilteredRecipes().length / itemsPerPage);
      currentPage = totalPages;
      renderRecipeTable();
    });

    document.getElementById('pageSelect').addEventListener('change', (e) => {
      currentPage = parseInt(e.target.value);
      renderRecipeTable();
    });

    // Featured: recipes with partner or cookoff tag, 6 most recent by Date column
    // Accepts numeric (MM-DD-YYYY or YYYY-MM-DD) or locale style (e.g. "Jan 1, 2026")
    function parseDateForSort(dateStr) {
      if (!dateStr || typeof dateStr !== 'string') return 0;
      var s = dateStr.trim();
      var parts = s.split(/[-/]/);
      if (parts.length === 3) {
        var mm = parseInt(parts[0], 10), dd = parseInt(parts[1], 10), yyyy = parseInt(parts[2], 10);
        if (parts[0].length === 4) { yyyy = parseInt(parts[0], 10); mm = parseInt(parts[1], 10); dd = parseInt(parts[2], 10); }
        if (yyyy && mm && dd) {
          var d = new Date(yyyy, mm - 1, dd);
          return isNaN(d.getTime()) ? 0 : d.getTime();
        }
      }
      var parsed = Date.parse(s);
      return (parsed !== undefined && !isNaN(parsed)) ? parsed : 0;
    }
    var FEATURED_SLOT_6_URL = 'getting_started/prompt_guide/reason_guide.html';
    function getFeaturedRecipes() {
      var hasPartnerOrCookoff = function(r) {
        var tags = r.tags && Array.isArray(r.tags) ? r.tags : [];
        return tags.some(function(t) {
          var x = (t || '').toLowerCase();
          return x === 'general:partner-recipe' || x === 'general:cookoff-recipe';
        });
      };
      var partnerOrCookoff = recipes.filter(hasPartnerOrCookoff);
      partnerOrCookoff.sort(function(a, b) { return parseDateForSort(b.date) - parseDateForSort(a.date); });
      var featured = partnerOrCookoff.slice(0, 5);
      if (featured.length < 5) {
        var featuredUrls = {};
        featured.forEach(function(r) { featuredUrls[r.url] = true; });
        featuredUrls[FEATURED_SLOT_6_URL] = true;
        var rest = recipes.filter(function(r) { return !featuredUrls[r.url]; });
        rest.sort(function(a, b) { return parseDateForSort(b.date) - parseDateForSort(a.date); });
        while (featured.length < 5 && rest.length > 0) {
          featured.push(rest.shift());
        }
      }
      var slot6 = recipes.find(function(r) { return r.url === FEATURED_SLOT_6_URL; });
      featured.push(slot6 || {
        name: 'Prompt Guide Cosmos Reason 2',
        url: FEATURED_SLOT_6_URL,
        category: 'Getting Started',
        workload: 'Prompt Guide',
        thumbnail: null
      });
      return featured;
    }
    function renderFeaturedRecipes() {
      var container = document.getElementById('featuredRecipesContainer');
      if (!container) return;
      var featured = getFeaturedRecipes();
      container.innerHTML = '';
      if (featured.length === 0) {
        container.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; padding: 2rem; color: #999;">No featured recipes at this time.</div>';
        return;
      }
      featured.forEach(function(recipe) {
        var a = document.createElement('a');
        a.href = recipe.url;
        a.className = 'recipe-card';
        var thumb;
        if (recipe.thumbnail) {
          thumb = document.createElement('img');
          thumb.className = 'recipe-thumbnail';
          thumb.src = recipe.thumbnail;
          thumb.alt = (recipe.name || 'Recipe').substring(0, 80);
          thumb.loading = 'lazy';
        } else {
          thumb = document.createElement('div');
          thumb.className = 'recipe-thumbnail';
          thumb.style.background = 'linear-gradient(135deg, #76b900 0%, #5f9300 100%)';
          thumb.style.display = 'flex';
          thumb.style.alignItems = 'center';
          thumb.style.justifyContent = 'center';
          thumb.style.color = 'white';
          thumb.style.fontSize = '1.5rem';
          thumb.style.fontWeight = '700';
          thumb.textContent = (recipe.name || '').charAt(0).toUpperCase() || '?';
        }
        a.appendChild(thumb);
        var content = document.createElement('div');
        content.className = 'recipe-content';
        var title = document.createElement('div');
        title.className = 'recipe-title';
        title.textContent = recipe.name || 'Recipe';
        content.appendChild(title);
        var desc = document.createElement('div');
        desc.className = 'recipe-description';
        desc.textContent = [recipe.category, recipe.workload].filter(Boolean).join(', ') || 'Recipe';
        content.appendChild(desc);
        a.appendChild(content);
        container.appendChild(a);
      });
    }

    // Load recipes from JSON and initialize
    async function loadRecipes() {
      try {
        const response = await fetch('recipes.json');
        if (!response.ok) {
          throw new Error(`Failed to load recipes: ${response.status}`);
        }
        recipes = await response.json();
        isLoading = false;
        renderRecipeTable();
        renderFeaturedRecipes();
      } catch (error) {
        console.error('Error loading recipes:', error);
        isLoading = false;
        renderFeaturedRecipes();
        // Show error message in table
        const tbody = document.getElementById('recipeTableBody');
        tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; padding: 2rem; color: #999;">Failed to load recipes. Please try refreshing the page.</td></tr>';
        document.getElementById('paginationInfo').textContent = '0 items';
      }
    }

    // Initial load
    loadRecipes();
  </script>
  </div><!-- End landing-page-wrapper -->
