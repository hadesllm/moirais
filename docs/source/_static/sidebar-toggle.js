// Sidebar toggle for the alabaster left nav.
//
// Three-state model:
//   - auto   : no class on <body>; CSS picks the default per viewport
//              (visible above 1180px, hidden below).
//   - shown  : <body class="hl-sidebar-shown"> — force visible.
//   - hidden : <body class="hl-sidebar-hidden"> — force hidden.
//
// Click toggles between the two FORCE states, anchored to whatever
// the user is currently seeing. Choice persists in localStorage.
//
// Placement: if the docs-hydrate hub strip (`nav.hl-docs-hub`) is
// present, we inject the toggle INTO it next to the menu + theme
// buttons so flexbox handles horizontal alignment uniformly.  If
// the hub strip isn't present (any page that hasn't been hydrated
// yet), fall back to a fixed-position button at the top-left.
// CSS distinguishes the two via `.is-in-hub` vs `.is-fixed`.

(function () {
    'use strict';

    const STORAGE_KEY  = 'hl.sidebar.intent';
    const ICON_HIDDEN  = '⫸';   // click to show
    const ICON_VISIBLE = '⫷';   // click to hide

    function readIntent() {
        try { return localStorage.getItem(STORAGE_KEY); } catch (_) { return null; }
    }

    function writeIntent(intent) {
        try {
            if (intent) localStorage.setItem(STORAGE_KEY, intent);
            else localStorage.removeItem(STORAGE_KEY);
        } catch (_) { /* storage disabled */ }
    }

    function isSidebarVisible() {
        const sb = document.querySelector('div.sphinxsidebar');
        if (!sb) return false;
        return getComputedStyle(sb).display !== 'none';
    }

    function applyIntent(intent, btn) {
        const body = document.body;
        body.classList.remove('hl-sidebar-hidden', 'hl-sidebar-shown');
        if (intent === 'hidden') body.classList.add('hl-sidebar-hidden');
        if (intent === 'shown')  body.classList.add('hl-sidebar-shown');
        if (btn) {
            const visible = isSidebarVisible();
            btn.textContent = visible ? ICON_VISIBLE : ICON_HIDDEN;
            btn.setAttribute('aria-pressed', String(!visible));
            btn.setAttribute('title', visible ? 'Hide sidebar' : 'Show sidebar');
        }
    }

    function buildButton(insideHub) {
        const btn = document.createElement('button');
        btn.id = 'hl-sidebar-toggle';
        btn.type = 'button';
        btn.setAttribute('aria-label', 'Toggle sidebar');
        // Class lets CSS style the button to match siblings when inside
        // the hub (round, 30x30, same border colors), or fall back to
        // its standalone fixed-position style when hub is absent.
        if (insideHub) {
            btn.className = 'hl-docs-hub-sidebar is-in-hub';
        } else {
            btn.className = 'is-fixed';
        }
        return btn;
    }

    function setSidebar(intent, btn) {
        applyIntent(intent, btn);
        writeIntent(intent);
    }

    function buildCloseButton(btn) {
        // Floating close button only matters at mobile widths; CSS
        // hides it elsewhere.  Click closes the drawer.  Mirrors the
        // toggle button's state machine.
        const close = document.createElement('button');
        close.id = 'hl-sidebar-close';
        close.type = 'button';
        close.setAttribute('aria-label', 'Close sidebar');
        close.title = 'Close sidebar';
        close.textContent = '✕';
        close.addEventListener('click', () => setSidebar('hidden', btn));
        document.body.appendChild(close);
    }

    function wireBackdropClose() {
        // Click on the dimmed backdrop (not on the drawer itself) to
        // close.  Pseudo-element ::before catches the click in CSS-
        // land via z-index, but the actual click event lands on the
        // body — detect by checking the click target is body-or-
        // backdrop and the drawer is currently shown.
        document.addEventListener('click', (e) => {
            if (!document.body.classList.contains('hl-sidebar-shown')) return;
            if (window.innerWidth > 600) return;  // mobile only
            const sb = document.querySelector('div.sphinxsidebar');
            if (!sb) return;
            // If the click is on the sidebar OR any of its children,
            // OR the toggle/close buttons, leave it alone.
            const t = e.target;
            if (sb.contains(t)) return;
            if (t.id === 'hl-sidebar-toggle' || t.closest('#hl-sidebar-toggle')) return;
            if (t.id === 'hl-sidebar-close') return;
            // Otherwise treat it as a backdrop tap.
            setSidebar('hidden', document.getElementById('hl-sidebar-toggle'));
        });
    }

    function wireEscClose(btn) {
        document.addEventListener('keydown', (e) => {
            if (e.key !== 'Escape') return;
            if (!document.body.classList.contains('hl-sidebar-shown')) return;
            setSidebar('hidden', btn);
        });
    }

    function init() {
        if (document.getElementById('hl-sidebar-toggle')) return;

        const hub = document.querySelector('nav.hl-docs-hub');
        const menu = hub ? hub.querySelector('.hl-docs-hub-menu') : null;
        const insideHub = hub != null;

        const btn = buildButton(insideHub);

        if (insideHub && menu) {
            hub.insertBefore(btn, menu);
        } else if (insideHub) {
            hub.insertBefore(btn, hub.firstChild);
        } else {
            document.body.appendChild(btn);
        }

        applyIntent(readIntent(), btn);

        btn.addEventListener('click', function () {
            const currentlyVisible = isSidebarVisible();
            const next = currentlyVisible ? 'hidden' : 'shown';
            setSidebar(next, btn);
        });

        // Mobile-only affordances to close the drawer once it's open:
        // explicit ✕ button, backdrop tap, Escape key.
        buildCloseButton(btn);
        wireBackdropClose();
        wireEscClose(btn);

        const mq = window.matchMedia('(max-width: 1180px)');
        const onMQ = () => applyIntent(readIntent(), btn);
        if (mq.addEventListener) mq.addEventListener('change', onMQ);
        else if (mq.addListener) mq.addListener(onMQ);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();
