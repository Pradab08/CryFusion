// ======= Load Navbar and Sidebar Components =======
document.addEventListener("DOMContentLoaded", () => {
  // Determine current page location to set correct paths
  const isInPages = window.location.pathname.includes('/pages/');
  const basePath = isInPages ? '../' : './';
  
  // Load Navbar
  fetch(basePath + 'components/navbar.html')
    .then(response => {
      if (!response.ok) throw new Error('Failed to load navbar');
      return response.text();
    })
    .then(data => {
      document.body.insertAdjacentHTML('afterbegin', data);
    })
    .catch(error => {
      console.error('Error loading navbar:', error);
      // Fallback navbar
      const fallbackNav = `
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary sticky-top">
          <div class="container-fluid">
            <a class="navbar-brand" href="${basePath}index.html">CryFusion</a>
            <div class="navbar-nav ms-auto">
              <a class="nav-link" href="${basePath}index.html">Dashboard</a>
              <a class="nav-link" href="${basePath}pages/cry_analysis.html">Cry Analysis</a>
              <a class="nav-link" href="${basePath}pages/health.html">Health</a>
              <a class="nav-link" href="${basePath}pages/gemini_chat.html">AI Chat</a>
            </div>
          </div>
        </nav>
      `;
      document.body.insertAdjacentHTML('afterbegin', fallbackNav);
    });

  // Load Sidebar
  fetch(basePath + 'components/sidebar.html')
    .then(response => {
      if (!response.ok) throw new Error('Failed to load sidebar');
      return response.text();
    })
    .then(data => {
      const wrapper = document.createElement('div');
      wrapper.innerHTML = data;
      document.body.insertBefore(wrapper, document.body.children[1]);
    })
    .catch(error => {
      console.error('Error loading sidebar:', error);
      // Fallback sidebar
      const fallbackSidebar = `
        <div class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse" id="sidebar">
          <div class="d-flex flex-column p-3">
            <ul class="nav nav-pills flex-column mb-auto">
              <li class="nav-item">
                <a href="${basePath}index.html" class="nav-link">Dashboard</a>
              </li>
              <li>
                <a href="${basePath}pages/cry_analysis.html" class="nav-link">Cry Analysis</a>
              </li>
              <li>
                <a href="${basePath}pages/health.html" class="nav-link">Health</a>
              </li>
              <li>
                <a href="${basePath}pages/gemini_chat.html" class="nav-link">Gemini Chat</a>
              </li>
            </ul>
          </div>
        </div>
      `;
      const sidebarContainer = document.getElementById('sidebar');
      if (sidebarContainer) {
        sidebarContainer.innerHTML = fallbackSidebar;
      }
    });
});
