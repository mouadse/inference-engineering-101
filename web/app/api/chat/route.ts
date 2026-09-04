const MODEL_ID = "google/medgemma-27b-it";
const DEFAULT_API_URL =
  "https://mouadse--medgemma-27b-vllm-server.us-east.modal.direct";
const MAX_REQUEST_BYTES = 4_300_000;

// Prompting follows Google's MedGemma guidance and technical report: the model
// is evaluated at temperature 0 with short role prompts ("You are a helpful
// medical assistant." for text, "You are a helpful radiology assistant." for
// images), and long prescriptive templates push it off its instruction tuning
// and measurably hurt medical QA. Both prompts below therefore stay short.
// The vision prompt is also modality-agnostic: this endpoint receives skin,
// eye, and pathology photos, not just chest X-rays.
const TEXT_SYSTEM_PROMPT = `You are a helpful medical assistant providing clinical decision support. Reason from the symptoms, history, and results the user shares. Structure the answer as: likely causes ranked by probability with key distinguishing features; red-flag signs needing urgent care; sensible next steps and what extra information would narrow it down. State uncertainty honestly. This is educational information only, not a diagnosis; advise correlation with a qualified clinician.`;

const VISION_SYSTEM_PROMPT = `You are a helpful radiology assistant. Begin directly with the evaluation, organized under Image type and quality, Observations, Impression, and Limitations / suggested follow-up. State modality, view, anatomical region, and technical quality; report only findings actually visible, with pertinent negatives. Give a concise impression; include a short differential ranked by likelihood only when an abnormality is present, and flag urgent red-flag findings prominently when present. Note limitations and follow-up. State uncertainty honestly. This is educational information only; advise correlation with a qualified clinician.`;

function hasImageContent(messages: ChatMessage[]): boolean {
  return messages.some(
    (message) =>
      Array.isArray(message.content) &&
      message.content.some(
        (part) =>
          !!part &&
          typeof part === "object" &&
          (part as Record<string, unknown>).type === "image_url",
      ),
  );
}

type ChatMessage = {
  role: "user" | "assistant";
  content: unknown;
};

function isChatMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Record<string, unknown>;
  return (
    (message.role === "user" || message.role === "assistant") &&
    (typeof message.content === "string" || Array.isArray(message.content))
  );
}

export const runtime = "nodejs";
export const maxDuration = 300;

export async function POST(request: Request) {
  const contentLength = Number(request.headers.get("content-length") ?? 0);
  if (contentLength > MAX_REQUEST_BYTES) {
    return Response.json(
      { error: "The image is too large. Choose a file smaller than 3 MB." },
      { status: 413 },
    );
  }

  let input: unknown;
  try {
    input = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON request." }, { status: 400 });
  }

  const messages =
    input && typeof input === "object"
      ? (input as Record<string, unknown>).messages
      : null;

  if (
    !Array.isArray(messages) ||
    messages.length === 0 ||
    messages.length > 24 ||
    !messages.every(isChatMessage)
  ) {
    return Response.json(
      { error: "Send between 1 and 24 valid chat messages." },
      { status: 400 },
    );
  }

  const includesImage = hasImageContent(messages);
  const baseUrl = (process.env.MEDGEMMA_API_URL ?? DEFAULT_API_URL).replace(
    /\/$/,
    "",
  );

  try {
    const upstream = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        model: MODEL_ID,
        stream: true,
        stream_options: { include_usage: true },
        temperature: 0,
        max_tokens: 1500,
        messages: [
          {
            role: "system",
            content: includesImage ? VISION_SYSTEM_PROMPT : TEXT_SYSTEM_PROMPT,
          },
          ...messages,
        ],
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(240_000),
    });

    if (!upstream.ok || !upstream.body) {
      const detail = await upstream.text();
      console.error("MedGemma upstream error:", upstream.status, detail);
      return Response.json(
        { error: "The model service is unavailable. Please try again." },
        { status: upstream.status || 502 },
      );
    }

    return new Response(upstream.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (error) {
    console.error("MedGemma proxy failed:", error);
    return Response.json(
      { error: "The model took too long to respond. Please try again." },
      { status: 504 },
    );
  }
}
