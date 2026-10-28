/* Safe, dependency-free Markdown rendering for streamed assistant messages. */
(function () {
    "use strict";

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function renderInline(value) {
        const codeTokens = [];
        let html = escapeHtml(value).replace(/`([^`]+)`/g, (_, code) => {
            const token = `\u0000CODE${codeTokens.length}\u0000`;
            codeTokens.push(`<code>${code}</code>`);
            return token;
        });

        html = html
            .replace(/\[([^\]]+)]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
            .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
            .replace(/__([^_]+)__/g, "<strong>$1</strong>")
            .replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>")
            .replace(/(?<!_)_([^_\n]+)_(?!_)/g, "<em>$1</em>");

        return html.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => codeTokens[Number(index)]);
    }

    function isTableSeparator(line) {
        return /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
    }

    function tableCells(line) {
        return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
    }

    function render(markdown) {
        const lines = String(markdown ?? "").replace(/\r\n?/g, "\n").split("\n");
        const output = [];
        let index = 0;
        let paragraph = [];

        function flushParagraph() {
            if (!paragraph.length) return;
            output.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
            paragraph = [];
        }

        while (index < lines.length) {
            const line = lines[index];

            if (/^\s*```/.test(line)) {
                flushParagraph();
                const language = line.replace(/^\s*```/, "").trim();
                const code = [];
                index += 1;
                while (index < lines.length && !/^\s*```/.test(lines[index])) {
                    code.push(lines[index]);
                    index += 1;
                }
                output.push(`<pre><code${language ? ` class="language-${escapeHtml(language)}"` : ""}>${escapeHtml(code.join("\n"))}</code></pre>`);
                if (index < lines.length) index += 1;
                continue;
            }

            if (index + 1 < lines.length && line.includes("|") && isTableSeparator(lines[index + 1])) {
                flushParagraph();
                const headers = tableCells(line);
                index += 2;
                const rows = [];
                while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
                    rows.push(tableCells(lines[index]));
                    index += 1;
                }
                output.push(
                    `<div class="markdown-table-wrap"><table><thead><tr>${headers.map((cell) => `<th>${renderInline(cell)}</th>`).join("")}</tr></thead>` +
                    `<tbody>${rows.map((cells) => `<tr>${headers.map((_, cellIndex) => `<td>${renderInline(cells[cellIndex] || "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`
                );
                continue;
            }

            const heading = /^(#{1,6})\s+(.+)$/.exec(line);
            if (heading) {
                flushParagraph();
                const level = heading[1].length;
                output.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
                index += 1;
                continue;
            }

            if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) {
                flushParagraph();
                output.push("<hr>");
                index += 1;
                continue;
            }

            if (/^>\s?/.test(line)) {
                flushParagraph();
                const quote = [];
                while (index < lines.length && /^>\s?/.test(lines[index])) {
                    quote.push(lines[index].replace(/^>\s?/, ""));
                    index += 1;
                }
                output.push(`<blockquote>${quote.map(renderInline).join("<br>")}</blockquote>`);
                continue;
            }

            const listMatch = /^\s*([-*+] |\d+\. )(.+)$/.exec(line);
            if (listMatch) {
                flushParagraph();
                const ordered = /^\s*\d+\. /.test(line);
                const items = [];
                while (index < lines.length) {
                    const itemMatch = /^\s*([-*+] |\d+\. )(.+)$/.exec(lines[index]);
                    if (!itemMatch || /^\s*\d+\. /.test(lines[index]) !== ordered) break;
                    items.push(`<li>${renderInline(itemMatch[2])}</li>`);
                    index += 1;
                }
                output.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
                continue;
            }

            if (!line.trim()) {
                flushParagraph();
                index += 1;
                continue;
            }

            paragraph.push(line);
            index += 1;
        }

        flushParagraph();
        return output.join("") || "";
    }

    window.ChatMarkdown = { render };
})();
