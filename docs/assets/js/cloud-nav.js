(function () {
  const cloudPagePattern = /\/getting_started\/cloud_platform(?:\/|\.html)?(?:#.*)?$/;
  const cloudParentMatcher = /cloud_platform(?:\/|\.html|\.md)?(?:$|[#/?])/;

  const removeGeneratedSubitems = () => {
    document.querySelectorAll(".cloud-nav-generated").forEach((el) => el.remove());
  };

  const collectCloudSections = (baseHref) => {
    return Array.from(document.querySelectorAll(".platform-section"))
      .map((section) => {
        const id = section.getAttribute("id");
        const heading = section.querySelector(".platform-header h2");
        const label = heading ? heading.textContent.trim() : id;
        if (!id || !label) return null;
        return { href: `${baseHref}#${id}`, label };
      })
      .filter(Boolean);
  };

  const insertSubitems = (parentItem, baseHref) => {
    const list = parentItem?.parentElement;
    if (!list) return;

    const subitems = collectCloudSections(baseHref);
    if (!subitems.length) return;

    const fragment = document.createDocumentFragment();
    subitems.forEach(({ href, label }) => {
      const li = document.createElement("li");
      li.className = "md-nav__item cloud-nav-subitem cloud-nav-generated";

      const link = document.createElement("a");
      link.className = "md-nav__link";
      link.setAttribute("href", href);

      const span = document.createElement("span");
      span.className = "md-ellipsis";
      span.textContent = label;

      link.appendChild(span);
      li.appendChild(link);
      fragment.appendChild(li);
    });

    list.insertBefore(fragment, parentItem.nextSibling);
  };

  const syncCloudNav = () => {
    const onCloudPage = cloudPagePattern.test(window.location.pathname);
    document.body.classList.toggle("cloud-nav-open", onCloudPage);

    removeGeneratedSubitems();

    document.querySelectorAll(".md-nav__link").forEach((link) => {
      const href = link.getAttribute("href") || "";
      const parent = link.closest(".md-nav__item");
      if (!parent || !cloudParentMatcher.test(href)) return;

      parent.classList.add("cloud-nav-target");

      if (!onCloudPage) return;

      const baseHref = (href.split("#")[0] || "").replace(/\.md$/, ".html");
      insertSubitems(parent, baseHref || "getting_started/cloud_platform.html");
    });
  };

  const init = () => window.requestAnimationFrame(syncCloudNav);

  if (window.document$) {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
