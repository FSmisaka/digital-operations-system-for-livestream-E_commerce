/**
 * 通用分页功能
 */
document.addEventListener("DOMContentLoaded", function () {
  initNewsPagination();
  initInvPagination();
  initForumPagination();
});

function initNewsPagination() {
  const newsContainer = document.querySelector(".news-list-container");
  if (!newsContainer) return;

  const newsCards = newsContainer.querySelectorAll(".news-card");
  if (newsCards.length === 0) return;

  const itemsPerPage = 5;
  const totalPages = Math.ceil(newsCards.length / itemsPerPage);
  let currentPage = 1;

  const pagination = document.querySelector(".news-pagination");
  if (!pagination) return;

  updatePagination(pagination, currentPage, totalPages, "news");
  showPage(newsCards, currentPage, itemsPerPage);

  pagination.addEventListener("click", function (e) {
    if (e.target.tagName === "A" || e.target.parentElement.tagName === "A") {
      e.preventDefault();

      const pageItem = e.target.closest(".page-item");
      if (!pageItem) return;

      if (pageItem.classList.contains("disabled")) return;

      if (pageItem.classList.contains("page-prev")) {
        if (currentPage > 1) {
          currentPage--;
        }
      } else if (pageItem.classList.contains("page-next")) {
        if (currentPage < totalPages) {
          currentPage++;
        }
      } else {
        const pageNum = parseInt(e.target.textContent);
        if (!isNaN(pageNum)) {
          currentPage = pageNum;
        }
      }

      updatePagination(pagination, currentPage, totalPages, "news");
      showPage(newsCards, currentPage, itemsPerPage);
      newsContainer.scrollIntoView({ behavior: "smooth" });
    }
  });
}

function initInvPagination() {
  const invContainer = document.querySelector(".inv-list-container");
  if (!invContainer) return;

  const invCards = invContainer.querySelectorAll(".inv-card");
  if (invCards.length === 0) return;

  const itemsPerPage = 4;
  const totalPages = Math.ceil(invCards.length / itemsPerPage);
  let currentPage = 1;

  const pagination = document.querySelector(".inv-pagination");
  if (!pagination) return;

  updatePagination(pagination, currentPage, totalPages, "inv");
  showPage(invCards, currentPage, itemsPerPage);

  pagination.addEventListener("click", function (e) {
    if (e.target.tagName === "A" || e.target.parentElement.tagName === "A") {
      e.preventDefault();

      const pageItem = e.target.closest(".page-item");
      if (!pageItem) return;

      if (pageItem.classList.contains("disabled")) return;

      if (pageItem.classList.contains("page-prev")) {
        if (currentPage > 1) {
          currentPage--;
        }
      } else if (pageItem.classList.contains("page-next")) {
        if (currentPage < totalPages) {
          currentPage++;
        }
      } else {
        const pageNum = parseInt(e.target.textContent);
        if (!isNaN(pageNum)) {
          currentPage = pageNum;
        }
      }

      updatePagination(pagination, currentPage, totalPages, "inv");
      showPage(invCards, currentPage, itemsPerPage);
      invContainer.scrollIntoView({ behavior: "smooth" });
    }
  });
}

function initForumPagination() {
  const topicContainer = document.querySelector(".card:not(.mb-4) .topic-list-container");
  if (!topicContainer) return;

  const topicCards = topicContainer.querySelectorAll(".list-group-item.topic-card");
  if (topicCards.length === 0) return;

  const itemsPerPage = 10;
  const totalPages = Math.ceil(topicCards.length / itemsPerPage);
  let currentPage = 1;

  const pagination = document.querySelector(".forum-pagination");
  if (!pagination) return;

  topicCards.forEach(card => {
    card.style.display = "none";
  });

  updatePagination(pagination, currentPage, totalPages, "forum");
  showPage(topicCards, currentPage, itemsPerPage);

  pagination.addEventListener("click", function (e) {
    if (e.target.tagName === "A" || e.target.parentElement.tagName === "A") {
      e.preventDefault();

      const pageItem = e.target.closest(".page-item");
      if (!pageItem) return;

      if (pageItem.classList.contains("disabled")) return;

      if (pageItem.classList.contains("page-prev")) {
        if (currentPage > 1) {
          currentPage--;
        }
      } else if (pageItem.classList.contains("page-next")) {
        if (currentPage < totalPages) {
          currentPage++;
        }
      } else {
        const pageNum = parseInt(e.target.textContent);
        if (!isNaN(pageNum)) {
          currentPage = pageNum;
        }
      }

      updatePagination(pagination, currentPage, totalPages, "forum");
      showPage(topicCards, currentPage, itemsPerPage);
      topicContainer.scrollIntoView({ behavior: "smooth" });
    }
  });
}

function showPage(items, page, itemsPerPage) {
  const startIndex = (page - 1) * itemsPerPage;
  const endIndex = Math.min(startIndex + itemsPerPage, items.length);

  items.forEach((item) => {
    item.classList.add('d-none'); 
    item.style.display = "none";  
  });

  // 显示当前页的项目
  for (let i = startIndex; i < endIndex; i++) {
    if (items[i]) {
      items[i].classList.remove('d-none'); 
      items[i].style.display = "flex";     
    }
  }
}

function updatePagination(pagination, currentPage, totalPages, type) {
  pagination.innerHTML = "";

  const prevItem = document.createElement("li");
  prevItem.className = `page-item page-prev ${currentPage === 1 ? "disabled" : ""}`;
  const prevLink = document.createElement("a");
  prevLink.className = "page-link";
  prevLink.href = "#";
  prevLink.textContent = "上一页";
  prevItem.appendChild(prevLink);
  pagination.appendChild(prevItem);

  const maxPageButtons = 5;
  const startPage = Math.max(1, currentPage - Math.floor(maxPageButtons / 2));
  const endPage = Math.min(totalPages, startPage + maxPageButtons - 1);

  for (let i = startPage; i <= endPage; i++) {
    const pageItem = document.createElement("li");
    pageItem.className = `page-item ${i === currentPage ? "active" : ""}`;
    const pageLink = document.createElement("a");
    pageLink.className = "page-link";
    pageLink.href = "#";
    pageLink.textContent = i;
    pageItem.appendChild(pageLink);
    pagination.appendChild(pageItem);
  }

  const nextItem = document.createElement("li");
  nextItem.className = `page-item page-next ${currentPage === totalPages ? "disabled" : ""}`;
  const nextLink = document.createElement("a");
  nextLink.className = "page-link";
  nextLink.href = "#";
  nextLink.textContent = "下一页";
  nextItem.appendChild(nextLink);
  pagination.appendChild(nextItem);

  const infoElement = document.querySelector(`.${type}-pagination-info`);
  if (infoElement) {
    if (type === "news") {
      const startItem = (currentPage - 1) * (5) + 1;
      const endItem = Math.min(
        currentPage * (5), 
        document.querySelectorAll(".news-card").length
      );
      const totalItems = document.querySelectorAll(".news-card").length;
      infoElement.textContent = `显示 ${startItem}-${endItem} 条，共 ${totalItems} 条`;
    }
    else if (type === "inv") {
      const startItem = (currentPage - 1) * (4) + 1;
      const endItem = Math.min(
        currentPage * (4), 
        document.querySelectorAll(".inv-card").length
      );
      const totalItems = document.querySelectorAll(".inv-card").length;
      infoElement.textContent = `显示 ${startItem}-${endItem} 条，共 ${totalItems} 条`;
    }
    else {
      const latestContainer = document.querySelector(".card:not(.mb-4) .topic-list-container");
      const totalItems = latestContainer.querySelectorAll(".list-group-item.topic-card").length;
      const startItem = (currentPage - 1) * (10) + 1;
      const endItem = Math.min(
        currentPage * (10), 
        totalItems
      );
      infoElement.textContent = `显示 ${startItem}-${endItem} 条，共 ${totalItems} 条`;
    }
  }
}
