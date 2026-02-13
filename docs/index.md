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

    /* Featured Recipes */
    .featured-recipes {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
  </style>
  
  <div class="landing-page-wrapper">
  <!-- Hero Section -->
  <div class="hero">
    <h1>NVIDIA Cosmos Cookbook</h1>
    <p>A practical guide with recipes to build, fine-tune, and deploy physical-AI</p>
  </div>

  <!-- Featured Recipes Section -->
  <div class="container">
    <div class="section-header">Featured recipes</div>
    
    <div class="featured-recipes">
      <a href="recipes/inference/reason2/intbot_showcase/inference.html" class="recipe-card">
        <img src="recipes/inference/reason2/intbot_showcase/assets/IntBot-GTC.jpg" alt="Egocentric Social and Physical Reasoning" class="recipe-thumbnail">
        <div class="recipe-content">
          <div class="recipe-title">Egocentric Social and Physical Reasoning with Cosmos-Reason2-8B</div>
          <div class="recipe-description">Robotics, Inference</div>
        </div>
      </a>

      <a href="recipes/post_training/reason2/video_caption_vqa/post_training.html" class="recipe-card">
        <img src="recipes/post_training/reason2/video_caption_vqa/assets/mcq_vqa_results.png" alt="Post-train Cosmos Reason 2 for AV Video Captioning & VQA" class="recipe-thumbnail">
        <div class="recipe-content">
          <div class="recipe-title">Post-train Cosmos Reason 2 for AV Video Captioning & VQA</div>
          <div class="recipe-description">Autonomous Vehicles, Post-training</div>
        </div>
      </a>

      <a href="recipes/post_training/reason1/physical-plausibility-check/post_training.html" class="recipe-card">
        <img src="recipes/post_training/reason1/physical-plausibility-check/assets/sft_accuracy.png" alt="Physical Plausibility Prediction with Cosmos Reason 1" class="recipe-thumbnail">
        <div class="recipe-content">
          <div class="recipe-title">Physical Plausibility Prediction with Cosmos Reason 1</div>
          <div class="recipe-description">Vision AI, Post-training</div>
        </div>
      </a>

      <a href="recipes/data_curation/embedding_analysis/embedding_analysis.html" class="recipe-card">
        <img src="recipes/data_curation/embedding_analysis/assets/clusters.png" alt="Curate data for Cosmos Predict Fine-Tuning using Cosmos Curator" class="recipe-thumbnail">
        <div class="recipe-content">
          <div class="recipe-title">Curate data for Cosmos Predict Fine-Tuning using Cosmos Curator</div>
          <div class="recipe-description">Vision AI, Curation</div>
        </div>
      </a>
    </div>

    <!-- All Recipes Section -->
    <div class="all-recipes-section">
      <div class="section-header">All recipes</div>
      
      <div class="filter-controls">
        <button class="filter-btn active" data-category="all">All</button>
        <button class="filter-btn" data-category="Robotics">Robotics</button>
        <button class="filter-btn" data-category="Autonomous Vehicles">Autonomous Vehicles</button>
        <button class="filter-btn" data-category="Vision AI">Vision AI</button>
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
    let currentCategory = 'all';
    let searchQuery = '';
    let isLoading = true;

    // Get filtered recipes
    function getFilteredRecipes() {
      return recipes.filter(recipe => {
        const matchesCategory = currentCategory === 'all' || recipe.category === currentCategory;
        const matchesSearch = searchQuery === '' || 
          recipe.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          recipe.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
          recipe.category.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesCategory && matchesSearch;
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
        row.innerHTML = `
          <td><a href="${recipe.url}">${recipe.name}</a></td>
          <td>${recipe.workload}</td>
          <td>${recipe.category}</td>
          <td>${recipe.date}</td>
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

    // Category filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentCategory = btn.dataset.category;
        currentPage = 1;
        renderRecipeTable();
      });
    });

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
      } catch (error) {
        console.error('Error loading recipes:', error);
        isLoading = false;
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