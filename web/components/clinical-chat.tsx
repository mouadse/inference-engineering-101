"use client";

import {
  AlertTriangle,
  ArrowUp,
  CircleStop,
  FileImage,
  ImagePlus,
  Plus,
  ScanLine,
  ShieldCheck,
  Stethoscope,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

const MAX_IMAGE_BYTES = 3_000_000;
const DEFAULT_IMAGE_PROMPT =
  "Review this X-ray systematically. Describe image quality, key observations, a concise impression, and important limitations.";

const STARTER_QUESTIONS = [
  "What are the warning signs of pneumonia?",
  "Help me understand these blood test results",
  "What should I ask at my next appointment?",
];

type ImageAttachment = {
  dataUrl: string;
  name: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  image?: ImageAttachment;
  status?: "streaming" | "complete";
};

function RadiographIllustration() {
  return (
    <div className="radiograph" aria-hidden="true">
      <svg viewBox="0 0 240 288" role="presentation">
        <rect className="radiograph-plate" x="10" y="8" width="220" height="272" rx="6" />
        <path className="radiograph-spine" d="M120 37v207" />
        <path
          className="radiograph-lung"
          d="M108 56C82 44 47 66 40 111c-8 53 9 111 48 125 17 6 28-11 27-32l-7-148Z"
        />
        <path
          className="radiograph-lung"
          d="M132 56c26-12 61 10 68 55 8 53-9 111-48 125-17 6-28-11-27-32l7-148Z"
        />
        <path className="radiograph-bone" d="M53 81c18-18 42-23 57-13M187 81c-18-18-42-23-57-13" />
        <path className="radiograph-bone" d="M43 108c22-20 48-26 68-15M197 108c-22-20-48-26-68-15" />
        <path className="radiograph-bone" d="M39 138c24-18 50-23 74-12M201 138c-24-18-50-23-74-12" />
        <path className="radiograph-bone" d="M42 168c24-15 48-18 72-7M198 168c-24-15-48-18-72-7" />
        <path className="radiograph-bone" d="M52 197c20-10 40-11 61-2M188 197c-20-10-40-11-61-2" />
        <path className="radiograph-heart" d="M120 145c-16-20-42-4-34 21 6 18 24 32 34 40 10-8 28-22 34-40 8-25-18-41-34-21Z" />
        <path className="radiograph-marker" d="M30 30h18M39 21v18" />
      </svg>
      <span>Vision enabled</span>
    </div>
  );
}

export function ClinicalChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [image, setImage] = useState<ImageAttachment | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  function handleImageFile(file: File | undefined) {
    setError(null);
    if (!file) return;
    if (!(["image/jpeg", "image/png", "image/webp"] as string[]).includes(file.type)) {
      setError("Choose a JPG, PNG, or WebP image.");
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setError("Choose an image smaller than 3 MB.");
      return;
    }

    const reader = new FileReader();
    reader.onerror = () => setError("That image could not be read. Try another file.");
    reader.onload = () => {
      if (typeof reader.result !== "string") return;
      setImage({ dataUrl: reader.result, name: file.name });
      textareaRef.current?.focus();
    };
    reader.readAsDataURL(file);
  }

  async function sendMessage() {
    const question = draft.trim() || (image ? DEFAULT_IMAGE_PROMPT : "");
    if (!question || isStreaming) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
      image: image ?? undefined,
      status: "complete",
    };
    const assistantId = crypto.randomUUID();
    const pendingMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      status: "streaming",
    };
    const history = [...messages, userMessage];
    const apiMessages = history.map((message) => {
      if (message.role === "assistant" || !message.image) {
        return { role: message.role, content: message.content };
      }
      return {
        role: "user" as const,
        content: [
          { type: "text", text: message.content },
          {
            type: "image_url",
            image_url: { url: message.image.dataUrl },
          },
        ],
      };
    });

    setMessages([...history, pendingMessage]);
    setDraft("");
    setImage(null);
    setError(null);
    setIsStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;
    let receivedText = false;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: apiMessages }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as {
          error?: string;
        } | null;
        throw new Error(body?.error ?? "The model service did not respond.");
      }
      if (!response.body) throw new Error("The model returned an empty response.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;

      while (!finished) {
        const { done, value } = await reader.read();
        finished = done;
        buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const event of events) {
          for (const line of event.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6);
            if (data === "[DONE]") continue;

            const chunk = JSON.parse(data) as {
              choices?: Array<{ delta?: { content?: string } }>;
            };
            const text = chunk.choices?.[0]?.delta?.content;
            if (!text) continue;
            receivedText = true;
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + text }
                  : message,
              ),
            );
          }
        }
      }

      if (!receivedText) throw new Error("The model returned an empty response.");
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? { ...message, status: "complete" } : message,
        ),
      );
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === "AbortError") {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  content: message.content || "Response stopped.",
                  status: "complete",
                }
              : message,
          ),
        );
      } else {
        setMessages((current) => current.filter((message) => message.id !== assistantId));
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Something went wrong. Please try again.",
        );
      }
    } finally {
      abortControllerRef.current = null;
      setIsStreaming(false);
    }
  }

  function startNewConsultation() {
    if (isStreaming) abortControllerRef.current?.abort();
    setMessages([]);
    setDraft("");
    setImage(null);
    setError(null);
    textareaRef.current?.focus();
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
          </div>
          <div>
            <p className="brand-name">MedGemma</p>
            <p className="brand-edition">Clinical workspace</p>
          </div>
        </div>

        <div className="model-status" aria-label="Model is online">
          <span className="status-dot" />
          <span className="status-model">MedGemma 27B · </span>Vision ready
        </div>

        <div className="sidebar-body">
          <div className="sidebar-intro">
            <p className="eyebrow">A considered second look</p>
            <h2>Bring the context. Keep the judgment.</h2>
            <p>
              Explore symptoms, prepare for appointments, or ask for a structured
              review of a medical image.
            </p>
          </div>

          <button className="new-consultation" type="button" onClick={startNewConsultation}>
            <Plus size={17} aria-hidden="true" />
            New consultation
          </button>

          <ol className="workflow" aria-label="How to use this workspace">
            <li>
              <span>01</span>
              <div>
                <strong>Share the context</strong>
                <p>Symptoms, timing, history, or your question.</p>
              </div>
            </li>
            <li>
              <span>02</span>
              <div>
                <strong>Add an image</strong>
                <p>JPG, PNG, or WebP—never a DICOM file.</p>
              </div>
            </li>
            <li>
              <span>03</span>
              <div>
                <strong>Verify with a clinician</strong>
                <p>Use the answer to ask better questions.</p>
              </div>
            </li>
          </ol>

          <div className="privacy-note">
            <ShieldCheck size={18} aria-hidden="true" />
            <p>
              <strong>Protect patient privacy.</strong>
              Remove names, dates of birth, and identifiers before upload.
            </p>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="topbar-kicker">Experimental medical AI</p>
            <p className="topbar-title">
              {messages.length > 0 ? "Consultation in progress" : "New consultation"}
            </p>
          </div>
          <button
            className="topbar-action"
            type="button"
            onClick={startNewConsultation}
            disabled={messages.length === 0 && !draft && !image}
          >
            <Plus size={16} aria-hidden="true" />
            <span>Start over</span>
          </button>
        </header>

        <section className="conversation" aria-live="polite" aria-busy={isStreaming}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-copy">
                <p className="eyebrow">Medical reasoning, with perspective</p>
                <h1>A clearer first read—not a final diagnosis.</h1>
                <p className="empty-lead">
                  Ask a health question or add an X-ray for a structured review. More
                  context leads to a more useful answer.
                </p>

                <div className="starter-list" aria-label="Suggested questions">
                  {STARTER_QUESTIONS.map((question, index) => (
                    <button
                      type="button"
                      key={question}
                      onClick={() => {
                        setDraft(question);
                        textareaRef.current?.focus();
                      }}
                    >
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      {question}
                      <ArrowUp size={15} aria-hidden="true" />
                    </button>
                  ))}
                </div>
              </div>
              <RadiographIllustration />
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article className={`message message-${message.role}`} key={message.id}>
                  <div className="message-heading">
                    {message.role === "assistant" ? (
                      <span className="assistant-seal" aria-hidden="true">
                        M
                      </span>
                    ) : null}
                    <span>{message.role === "assistant" ? "MedGemma" : "You"}</span>
                  </div>
                  <div className="message-body">
                    {message.image ? (
                      <figure className="submitted-image">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={message.image.dataUrl} alt={`Uploaded medical image: ${message.image.name}`} />
                        <figcaption>
                          <ScanLine size={14} aria-hidden="true" />
                          {message.image.name}
                        </figcaption>
                      </figure>
                    ) : null}
                    {message.role === "assistant" ? (
                      <div className="markdown">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                        {message.status === "streaming" ? (
                          <span className="stream-cursor" aria-label="Generating response" />
                        ) : null}
                      </div>
                    ) : (
                      <p>{message.content}</p>
                    )}
                  </div>
                </article>
              ))}
              <div className="response-disclaimer">
                <Stethoscope size={16} aria-hidden="true" />
                Check important decisions with a qualified medical professional.
              </div>
              <div ref={conversationEndRef} />
            </div>
          )}
        </section>

        <div
          className={`composer-region${isDragging ? " is-dragging" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
              setIsDragging(false);
            }
          }}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            handleImageFile(event.dataTransfer.files[0]);
          }}
        >
          <div className="composer-wrap">
            {image ? (
              <div className="image-preview">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={image.dataUrl} alt={`Preview of ${image.name}`} />
                <div>
                  <span>Image attached</span>
                  <strong>{image.name}</strong>
                </div>
                <button type="button" onClick={() => setImage(null)} aria-label="Remove attached image">
                  <X size={17} aria-hidden="true" />
                </button>
              </div>
            ) : null}

            {error ? (
              <div className="error-banner" role="alert">
                <AlertTriangle size={17} aria-hidden="true" />
                <span>{error}</span>
                <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">
                  <X size={15} aria-hidden="true" />
                </button>
              </div>
            ) : null}

            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault();
                void sendMessage();
              }}
            >
              <input
                ref={fileInputRef}
                className="visually-hidden"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(event) => {
                  handleImageFile(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
              <button
                className="attach-button"
                type="button"
                onClick={() => fileInputRef.current?.click()}
                aria-label="Attach an X-ray or medical image"
                title="Attach an image"
              >
                <ImagePlus size={21} aria-hidden="true" />
              </button>
              <textarea
                ref={textareaRef}
                value={draft}
                rows={1}
                maxLength={4_000}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault();
                    void sendMessage();
                  }
                }}
                placeholder={image ? "What would you like to know about this image?" : "Ask a medical question…"}
                aria-label="Medical question"
              />
              {isStreaming ? (
                <button
                  className="send-button stop-button"
                  type="button"
                  onClick={() => abortControllerRef.current?.abort()}
                  aria-label="Stop response"
                >
                  <CircleStop size={20} aria-hidden="true" />
                </button>
              ) : (
                <button
                  className="send-button"
                  type="submit"
                  disabled={!draft.trim() && !image}
                  aria-label="Send question"
                >
                  <ArrowUp size={20} aria-hidden="true" />
                </button>
              )}
            </form>

            <div className="composer-meta">
              <span>
                <FileImage size={14} aria-hidden="true" />
                JPG, PNG, WebP · 3 MB max
              </span>
              <span className="keyboard-hint">Enter to send · Shift + Enter for a new line</span>
            </div>
          </div>
          <p className="medical-notice">
            Not for diagnosis or emergencies. If symptoms are severe or sudden, seek urgent care.
          </p>
          {isDragging ? (
            <div className="drop-overlay" aria-hidden="true">
              <ScanLine size={28} />
              <span>Drop the medical image here</span>
            </div>
          ) : null}
        </div>
      </main>
    </div>
  );
}
