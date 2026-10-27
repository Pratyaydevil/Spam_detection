const API_URL = "http://127.0.0.1:8000/analyze";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type !== "analyze-email") {
        return false;
    }

    fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(message.payload)
    })
        .then(async (response) => {
            const body = await response.json().catch(() => null);
            if (!response.ok) {
                const detail = body?.detail || `API returned HTTP ${response.status}`;
                throw new Error(detail);
            }
            sendResponse({ ok: true, result: body });
        })
        .catch((error) => {
            const detail = error instanceof TypeError
                ? "Could not reach the local API. Start the project with start.bat and try again."
                : error.message;
            sendResponse({ ok: false, error: detail });
        });

    return true;
});
