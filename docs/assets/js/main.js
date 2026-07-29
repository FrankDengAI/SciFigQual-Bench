(() => {
  const BASE = document.querySelector('base')?.getAttribute("href") || "./";

  // Resolve asset URLs relative to this page (works on GitHub Pages project sites)
  const resolve = (path) => {
    try {
      return new URL(path, window.location.href).href;
    } catch {
      return path;
    }
  };

  // Smooth-scroll for in-page anchors with sticky nav offset
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener("click", (e) => {
      const id = a.getAttribute("href");
      if (!id || id === "#") return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      history.pushState(null, "", id);
      document.getElementById("nav-links")?.classList.remove("open");
    });
  });

  // Mobile nav
  const toggle = document.getElementById("nav-toggle");
  const links = document.getElementById("nav-links");
  toggle?.addEventListener("click", () => links?.classList.toggle("open"));

  // Active section highlighting
  const sections = [...document.querySelectorAll("section[id]")];
  const navAnchors = [...document.querySelectorAll(".nav-links a[href^='#']")];
  const setActive = () => {
    const y = window.scrollY + 90;
    let current = sections[0]?.id;
    for (const s of sections) {
      if (s.offsetTop <= y) current = s.id;
    }
    navAnchors.forEach((a) => {
      a.classList.toggle("is-active", a.getAttribute("href") === `#${current}`);
    });
  };
  window.addEventListener("scroll", setActive, { passive: true });
  setActive();

  // Reveal on scroll
  const revealEls = document.querySelectorAll(".reveal");
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -36px 0px" }
  );
  revealEls.forEach((el) => io.observe(el));

  // Counters
  const format = (n, raw) => {
    if (/k/i.test(raw)) {
      const v = n / 1000;
      return (Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1).replace(/\.0$/, "")) + "k";
    }
    if (raw.includes("%")) return `${Math.round(n)}%`;
    if (raw.includes(".")) {
      const decimals = (raw.split(".")[1] || "").length;
      return n.toFixed(decimals);
    }
    return Math.round(n).toLocaleString("en-US");
  };

  const animateCount = (el) => {
    const raw = el.getAttribute("data-count") || "0";
    const target = parseFloat(raw.replace(/[^\d.]/g, ""));
    if (!Number.isFinite(target)) return;
    const duration = 1150;
    const start = performance.now();
    const tick = (t) => {
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = format(target * eased, raw);
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = format(target, raw);
    };
    requestAnimationFrame(tick);
  };

  const cio = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          animateCount(e.target);
          cio.unobserve(e.target);
        }
      });
    },
    { threshold: 0.45 }
  );
  document.querySelectorAll("[data-count]").forEach((el) => cio.observe(el));

  // Image load guards
  document.querySelectorAll("img[data-zoom], img[data-check]").forEach((img) => {
    img.addEventListener("error", () => {
      const wrap = document.createElement("div");
      wrap.className = "img-broken";
      wrap.textContent = `Image failed to load: ${img.getAttribute("src")}`;
      img.replaceWith(wrap);
    });
  });

  // Lightbox
  const lb = document.getElementById("lightbox");
  const lbImg = document.getElementById("lightbox-img");
  const closeLb = () => lb?.classList.remove("open");

  document.querySelectorAll("[data-zoom]").forEach((img) => {
    img.addEventListener("click", () => {
      if (!lb || !lbImg) return;
      lbImg.src = resolve(img.currentSrc || img.src);
      lbImg.alt = img.alt || "";
      lb.classList.add("open");
    });
  });
  lb?.addEventListener("click", (e) => {
    if (e.target === lb || e.target.classList.contains("lightbox-close")) closeLb();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLb();
  });

  // Copy citation
  const btn = document.getElementById("copy-cite");
  const pre = document.getElementById("citation");
  btn?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText((pre?.textContent || "").trim());
      btn.textContent = "Copied";
      btn.classList.add("ok");
      setTimeout(() => {
        btn.textContent = "Copy";
        btn.classList.remove("ok");
      }, 1500);
    } catch {
      btn.textContent = "Select text";
    }
  });

  // Quiet unused lint
  void BASE;
})();
