(() => {
    const hostId = "local-email-defender-extension";
    if (document.getElementById(hostId)) {
        return;
    }

    const host = document.createElement("div");
    host.id = hostId;
    document.documentElement.appendChild(host);
    const shadow = host.attachShadow({ mode: "open" });

    const style = document.createElement("style");
    style.textContent = `
        :host { all: initial; }
        .launcher {
            position: fixed; right: 24px; bottom: 24px; z-index: 2147483647;
            border: 0; border-radius: 999px; padding: 12px 18px;
            background: #6d28d9; color: white; cursor: pointer;
            font: 600 14px/1.2 Arial, sans-serif;
            box-shadow: 0 4px 16px #0004;
        }
        .launcher:hover { background: #5b21b6; }
        .panel {
            position: fixed; right: 24px; bottom: 78px; z-index: 2147483647;
            width: min(360px, calc(100vw - 48px)); max-height: 60vh; overflow: auto;
            box-sizing: border-box; padding: 18px; border: 1px solid #ddd;
            border-radius: 12px; background: #fff; color: #202124;
            box-shadow: 0 8px 30px #0003; font: 14px/1.5 Arial, sans-serif;
        }
        .panel[hidden] { display: none; }
        .panel h2 { margin: 0 0 8px; font-size: 17px; }
        .panel p { margin: 8px 0; overflow-wrap: anywhere; }
        .panel ul { margin: 8px 0; padding-left: 20px; }
        .panel .notice { color: #5f6368; font-size: 12px; }
        .panel .error { color: #b3261e; }
        .close {
            float: right; border: 0; background: transparent; cursor: pointer;
            color: #5f6368; font-size: 20px; line-height: 1;
        }
    `;

    const button = document.createElement("button");
    button.className = "launcher";
    button.type = "button";
    button.textContent = "Analyze open email";

    const panel = document.createElement("section");
    panel.className = "panel";
    panel.hidden = true;
    panel.setAttribute("role", "status");
    panel.setAttribute("aria-live", "polite");

    shadow.append(style, button, panel);

    function showMessage(title, message, isError = false) {
        panel.replaceChildren();
        const close = document.createElement("button");
        close.className = "close";
        close.type = "button";
        close.textContent = "×";
        close.setAttribute("aria-label", "Close analysis");
        close.addEventListener("click", () => {
            panel.hidden = true;
        });
        const heading = document.createElement("h2");
        heading.textContent = title;
        const text = document.createElement("p");
        text.textContent = message;
        if (isError) {
            text.className = "error";
        }
        panel.append(close, heading, text);
        panel.hidden = false;
    }

    function currentEmailPayload() {
        const bodyElements = Array.from(document.querySelectorAll(
            '[role="main"] .a3s.aiL, [role="main"] .a3s'
        ));
        const visibleBodies = bodyElements.filter((element) =>
            element.getClientRects().length > 0 && element.innerText.trim()
        );
        const emailBody = visibleBodies.at(-1);
        if (!emailBody) {
            throw new Error("Open a Gmail message before starting an analysis.");
        }

        const text = emailBody.innerText.trim();
        const links = Array.from(emailBody.querySelectorAll('a[href]'))
            .map((anchor) => {
                try {
                    const url = new URL(anchor.href);
                    return url.protocol === "http:" || url.protocol === "https:" ? url : null;
                } catch {
                    return null;
                }
            })
            .filter((url) => url !== null);
        const urlLength = links.length
            ? links.reduce((total, url) => total + url.href.length, 0) / links.length
            : 0;
        const subdomainCount = links.reduce((maximum, url) => {
            const labels = url.hostname.split(".");
            return Math.max(maximum, Math.max(0, labels.length - 2));
        }, 0);

        return {
            text,
            url_length: urlLength,
            subdomain_count: subdomainCount,
            https_presence: links.length && links.every((url) => url.protocol === "https:") ? 1 : 0,
            spf_dkim_failure: 0,
            sender_reputation: 0.5,
            recipient_count: 1
        };
    }

    function displayResult(result) {
        panel.replaceChildren();
        const close = document.createElement("button");
        close.className = "close";
        close.type = "button";
        close.textContent = "×";
        close.setAttribute("aria-label", "Close analysis");
        close.addEventListener("click", () => {
            panel.hidden = true;
        });

        const heading = document.createElement("h2");
        heading.textContent = `Result: ${result.classification}`;
        const uncertainty = document.createElement("p");
        uncertainty.textContent = `Prediction uncertainty (not calibrated): ${Number(result.epistemic_uncertainty).toFixed(3)}${result.requires_human_review ? " — human review recommended" : ""}`;
        const probabilityTitle = document.createElement("p");
        probabilityTitle.textContent = "Class probabilities:";
        const probabilities = document.createElement("ul");
        for (const [label, value] of Object.entries(result.probabilities || {})) {
            const item = document.createElement("li");
            item.textContent = `${label}: ${(Number(value) * 100).toFixed(1)}%`;
            probabilities.appendChild(item);
        }
        const notice = document.createElement("p");
        notice.className = "notice";
        notice.textContent = "A text-only model trained on public email datasets produced this advisory prediction; it is not a guarantee of safety. Analysis is sent only to the API on this computer.";
        panel.append(close, heading, uncertainty, probabilityTitle, probabilities, notice);
        panel.hidden = false;
    }

    button.addEventListener("click", async () => {
        button.disabled = true;
        try {
            const payload = currentEmailPayload();
            showMessage("Analyzing email…", "Sending this open message to the local API on your computer.");
            const response = await chrome.runtime.sendMessage({
                type: "analyze-email",
                payload
            });
            if (!response?.ok) {
                throw new Error(response?.error || "The local analysis request failed.");
            }
            displayResult(response.result);
        } catch (error) {
            showMessage("Analysis unavailable", error.message || "An unexpected error occurred.", true);
        } finally {
            button.disabled = false;
        }
    });
})();
