document.addEventListener("DOMContentLoaded", function () {
  activateCurrentNavLink();
  addCardHoverEffects();
  initializeTooltips();
  addSmoothScrolling();
  addBackToTopButton();
});

function activateCurrentNavLink() {
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll(".navbar-nav .nav-link");

  navLinks.forEach((link) => {
    link.classList.remove("active");
    const href = link.getAttribute("href");
    if (href && currentPath.includes(href) && href !== "/") {
      link.classList.add("active");
    } else if (currentPath === "/" && href === "/") {
      link.classList.add("active");
    }
  });
}

function addCardHoverEffects() {
  const cards = document.querySelectorAll(".card");

  cards.forEach((card) => {
    card.addEventListener("mouseenter", function () {
      this.style.transform = "translateY(-5px)";
      this.style.boxShadow = "0 10px 20px rgba(0,0,0,0.1)";
    });

    card.addEventListener("mouseleave", function () {
      this.style.transform = "translateY(0)";
      this.style.boxShadow = "0 4px 6px rgba(0,0,0,0.05)";
    });
  });
}

function initializeTooltips() {
  if (
    typeof bootstrap !== "undefined" &&
    typeof bootstrap.Tooltip !== "undefined"
  ) {
    const tooltipTriggerList = [].slice.call(
      document.querySelectorAll('[data-bs-toggle="tooltip"]'),
    );
    tooltipTriggerList.map(function (tooltipTriggerEl) {
      return new bootstrap.Tooltip(tooltipTriggerEl);
    });
  }
}

function addSmoothScrolling() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();

      const targetId = this.getAttribute("href");
      if (targetId === "#") return;

      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: "smooth",
        });
      }
    });
  });
}

function addBackToTopButton() {
  const backToTopBtn = document.createElement("button");
  backToTopBtn.innerHTML = '<i class="bi bi-arrow-up"></i>';
  backToTopBtn.className = "btn btn-primary back-to-top";
  backToTopBtn.style.position = "fixed";
  backToTopBtn.style.bottom = "20px";
  backToTopBtn.style.right = "20px";
  backToTopBtn.style.display = "none";
  backToTopBtn.style.zIndex = "1000";
  backToTopBtn.style.width = "40px";
  backToTopBtn.style.height = "40px";
  backToTopBtn.style.borderRadius = "50%";
  backToTopBtn.style.padding = "0";

  document.body.appendChild(backToTopBtn);

  window.addEventListener("scroll", function () {
    if (window.pageYOffset > 300) {
      backToTopBtn.style.display = "block";
    } else {
      backToTopBtn.style.display = "none";
    }
  });

  backToTopBtn.addEventListener("click", function () {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  });
}

function formatNumber(number) {
  return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatDate(dateString) {
  const date = new Date(dateString);
  return date.toLocaleDateString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
}

function calculatePercentChange(oldValue, newValue) {
  return (((newValue - oldValue) / oldValue) * 100).toFixed(2) + "%";
}

function showLoading(elementId) {
  const element = document.getElementById(elementId);
  if (element) {
    element.innerHTML =
      '<div class="text-center p-3"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">加载中...</span></div><p class="mt-2">加载中...</p></div>';
  }
}

function hideLoading(elementId) {
  const element = document.getElementById(elementId);
  if (element) {
    element.innerHTML = "";
  }
}

function showNotification(message, type = "success") {
  if (
    typeof bootstrap !== "undefined" &&
    typeof bootstrap.Toast !== "undefined"
  ) {
    const toastElement = document.createElement("div");
    toastElement.className = `toast align-items-center text-white bg-${type} border-0`;
    toastElement.setAttribute("role", "alert");
    toastElement.setAttribute("aria-live", "assertive");
    toastElement.setAttribute("aria-atomic", "true");

    toastElement.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;

    // 创建Toast容器
    let toastContainer = document.querySelector(".toast-container");
    if (!toastContainer) {
      toastContainer = document.createElement("div");
      toastContainer.className =
        "toast-container position-fixed bottom-0 end-0 p-3";
      document.body.appendChild(toastContainer);
    }

    // 添加Toast到容器
    toastContainer.appendChild(toastElement);

    // 初始化并显示Toast
    const toast = new bootstrap.Toast(toastElement);
    toast.show();

    // 自动移除Toast元素
    toastElement.addEventListener("hidden.bs.toast", function () {
      toastElement.remove();
    });
  } else {
    // 如果没有Bootstrap的Toast功能，使用alert
    alert(message);
  }
}
