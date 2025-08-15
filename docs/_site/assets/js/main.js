// Adaptive Multi-Fidelity Aerospace Optimization - Website JavaScript
// Minimal JavaScript for enhanced user experience

document.addEventListener('DOMContentLoaded', function() {
    
    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add active class to current navigation item
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        const linkPath = new URL(link.href).pathname;
        if (linkPath === currentPath || (currentPath === '/' && linkPath.endsWith('/'))) {
            link.classList.add('active');
        }
    });
    
    // Mobile navigation toggle (if needed)
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('nav-menu--open');
            navToggle.classList.toggle('nav-toggle--active');
        });
    }
    
    // Performance counter animation
    const performanceCounters = document.querySelectorAll('.stat-value, .hero-stat-value, .result-value');
    
    function animateCounter(element) {
        const target = element.textContent;
        const numericValue = parseFloat(target.replace(/[^\d.]/g, ''));
        
        if (!isNaN(numericValue)) {
            let current = 0;
            const increment = numericValue / 50;
            const timer = setInterval(() => {
                current += increment;
                if (current >= numericValue) {
                    current = numericValue;
                    clearInterval(timer);
                }
                
                // Format the number back with original suffix
                let formatted = current.toFixed(1);
                if (target.includes('%')) {
                    formatted += '%';
                }
                element.textContent = formatted;
            }, 20);
        }
    }
    
    // Intersection Observer for counter animation
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('animated')) {
                entry.target.classList.add('animated');
                animateCounter(entry.target);
            }
        });
    }, observerOptions);
    
    performanceCounters.forEach(counter => {
        observer.observe(counter);
    });
    
    // Image lazy loading fallback
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
    
    // Copy code blocks functionality
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.addEventListener('click', () => {
            navigator.clipboard.writeText(block.textContent).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        
        const pre = block.parentElement;
        pre.style.position = 'relative';
        pre.appendChild(button);
    });
    
    // External link handling
    const externalLinks = document.querySelectorAll('a[href^="http"]');
    externalLinks.forEach(link => {
        if (!link.href.includes(window.location.hostname)) {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
        }
    });
    
    // Performance metrics highlighting
    const performanceMetrics = {
        'cost_reduction': '85.7%',
        'solution_accuracy': '99.5%',
        'test_coverage': '100%'
    };
    
    // Highlight performance achievements
    Object.keys(performanceMetrics).forEach(metric => {
        const elements = document.querySelectorAll(`[data-metric="${metric}"]`);
        elements.forEach(element => {
            element.addEventListener('mouseenter', () => {
                element.style.backgroundColor = '#fff3cd';
                element.style.transition = 'background-color 0.3s ease';
            });
            
            element.addEventListener('mouseleave', () => {
                element.style.backgroundColor = 'transparent';
            });
        });
    });
    
    // Search functionality (basic)
    const searchInput = document.querySelector('#search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            const content = document.querySelectorAll('.page-content p, .page-content h1, .page-content h2, .page-content h3');
            
            content.forEach(element => {
                const text = element.textContent.toLowerCase();
                if (query && text.includes(query)) {
                    element.style.backgroundColor = '#fff3cd';
                } else {
                    element.style.backgroundColor = 'transparent';
                }
            });
        });
    }
    
    // Print styles trigger
    const printButton = document.querySelector('#print-page');
    if (printButton) {
        printButton.addEventListener('click', () => {
            window.print();
        });
    }
    
    // Page load performance tracking
    window.addEventListener('load', () => {
        const loadTime = performance.now();
        console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
        
        // Optional: Send analytics if configured
        if (typeof gtag !== 'undefined') {
            gtag('event', 'page_load_time', {
                'event_category': 'Performance',
                'event_label': 'Load Time',
                'value': Math.round(loadTime)
            });
        }
    });
    
});