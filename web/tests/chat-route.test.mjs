import assert from "node:assert/strict";
import test from "node:test";

import { POST } from "../app/api/chat/route.ts";

test("text-only requests receive the clinical system prompt", async (t) => {
  const originalFetch = globalThis.fetch;
  let upstreamRequest;

  globalThis.fetch = async (_url, init) => {
    upstreamRequest = JSON.parse(init.body);
    return new Response("data: [DONE]\n\n", {
      headers: { "Content-Type": "text/event-stream" },
    });
  };
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  const response = await POST(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [
          {
            role: "user",
            content:
              "A patient has hypertension, hypokalemia, metabolic alkalosis, suppressed renin, and suppressed aldosterone. What is the most likely diagnosis?",
          },
        ],
      }),
    }),
  );

  assert.equal(response.status, 200);
  assert.equal(upstreamRequest.model, "google/medgemma-27b-it");
  assert.equal(upstreamRequest.temperature, 0);
  assert.equal(upstreamRequest.messages.length, 2);
  assert.equal(upstreamRequest.messages[0].role, "system");
  assert.match(upstreamRequest.messages[0].content, /helpful medical assistant/);
  assert.match(upstreamRequest.messages[0].content, /red-flag/i);
  assert.deepEqual(upstreamRequest.messages[1], {
    role: "user",
    content:
      "A patient has hypertension, hypokalemia, metabolic alkalosis, suppressed renin, and suppressed aldosterone. What is the most likely diagnosis?",
  });
});

test("requests with an attachment receive image-analysis instructions", async (t) => {
  const originalFetch = globalThis.fetch;
  let upstreamRequest;

  globalThis.fetch = async (_url, init) => {
    upstreamRequest = JSON.parse(init.body);
    return new Response("data: [DONE]\n\n", {
      headers: { "Content-Type": "text/event-stream" },
    });
  };
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  const response = await POST(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Review this radiograph." },
              {
                type: "image_url",
                image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
              },
            ],
          },
        ],
      }),
    }),
  );

  assert.equal(response.status, 200);
  assert.equal(upstreamRequest.temperature, 0);
  assert.match(
    upstreamRequest.messages[0].content,
    /Image type and quality, Observations, Impression/,
  );
  assert.match(upstreamRequest.messages[0].content, /only findings actually visible/);
});
