// Smart clipboard for code blocks — strips `# comment` lines on copy.
//
// MOIRAIS docs include shell + Python snippets where lines are
// annotated with comments, e.g.
//
//   pip install moirais        # Python package (41 datasets)
//   pip install moirais[interactive]  # + Terminal IDE
//
// New users paste those into a shell and either (a) get an error
// because the shell doesn't know what `# Python package (41 datasets)`
// is, or (b) the comment runs as part of the command.
//
// Two affordances ship here:
//
//   1. A small "Copy" button in the top-right of every <pre> block
//      that copies the stripped version (no comments).
//
//   2. A native copy-event interception: if the user uses
//      Cmd/Ctrl+C with a selection that's fully inside a <pre>,
//      the clipboard text is rewritten on the fly.
//
// Strip rules (deliberately conservative):
//   - Pure-comment lines (matching ^\s*#) are removed entirely.
//   - Inline ` # ` (space-hash-space) is the conventional inline-
//     comment marker; everything from that point to end-of-line is
//     stripped.
//   - We do NOT try to parse strings or AST.  ` # ` inside a
//     string literal is rare in copy-paste install snippets.
//
// Behaviour is announced via a small floating "✂ comments stripped"
// toast so the user knows what changed.

(function () {
    'use strict';

    function stripComments(text) {
        const lines = text.split('\n');
        const out = [];
        for (const raw of lines) {
            if (/^\s*#/.test(raw)) continue;          // whole-line comment
            let line = raw;
            const idx = line.indexOf(' # ');
            if (idx >= 0) line = line.substring(0, idx);
            line = line.replace(/\s+$/, '');           // trim trailing space
            out.push(line);
        }
        // Collapse leading + trailing blank lines.
        while (out.length && out[0] === '') out.shift();
        while (out.length && out[out.length - 1] === '') out.pop();
        return out.join('\n');
    }

    let toastEl = null;
    function toast(msg) {
        if (!toastEl) {
            toastEl = document.createElement('div');
            toastEl.className = 'hl-clipboard-toast';
            document.body.appendChild(toastEl);
        }
        toastEl.textContent = msg;
        toastEl.classList.add('is-visible');
        clearTimeout(toast._t);
        toast._t = setTimeout(() => toastEl.classList.remove('is-visible'), 1600);
    }

    function copyToClipboard(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(text);
        }
        // Fallback for older browsers / non-secure contexts.
        return new Promise((resolve, reject) => {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            document.body.appendChild(ta);
            ta.select();
            try {
                document.execCommand('copy');
                resolve();
            } catch (e) {
                reject(e);
            } finally {
                document.body.removeChild(ta);
            }
        });
    }

    function attachButton(pre) {
        if (pre.querySelector('.hl-copy-btn')) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'hl-copy-btn';
        btn.textContent = 'Copy';
        btn.setAttribute('aria-label', 'Copy code (strips # comments)');
        btn.title = 'Copy — # comments are removed';
        btn.addEventListener('click', function (e) {
            e.preventDefault();
            const original = pre.innerText;
            const cleaned = stripComments(original);
            copyToClipboard(cleaned).then(
                () => {
                    btn.textContent = '✓ Copied';
                    toast(cleaned !== original
                        ? '✂ comments stripped'
                        : 'copied');
                    setTimeout(() => { btn.textContent = 'Copy'; }, 1400);
                },
                () => { btn.textContent = '✗ Failed'; }
            );
        });
        pre.appendChild(btn);
        // Make sure pre has positioning context for the absolute btn.
        const pos = getComputedStyle(pre).position;
        if (pos === 'static') pre.style.position = 'relative';
    }

    function attachAll() {
        document.querySelectorAll('div.highlight pre, pre.literal-block')
            .forEach(attachButton);
    }

    // Native Cmd/Ctrl+C interception for selections inside a <pre>.
    function isInsideCodeBlock(node) {
        while (node && node !== document.body) {
            if (node.nodeType === 1 &&
                (node.matches?.('pre') ||
                 node.matches?.('div.highlight'))) {
                return true;
            }
            node = node.parentNode;
        }
        return false;
    }

    document.addEventListener('copy', function (e) {
        const sel = window.getSelection();
        if (!sel || sel.isCollapsed) return;
        if (!isInsideCodeBlock(sel.anchorNode) ||
            !isInsideCodeBlock(sel.focusNode)) return;
        const text = sel.toString();
        const cleaned = stripComments(text);
        if (cleaned === text) return; // no comments — let native copy run
        e.preventDefault();
        if (e.clipboardData) {
            e.clipboardData.setData('text/plain', cleaned);
        } else {
            // Old browsers: fall back to async write
            copyToClipboard(cleaned);
        }
        toast('✂ comments stripped');
    });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachAll, { once: true });
    } else {
        attachAll();
    }
})();
