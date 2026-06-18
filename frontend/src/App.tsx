import { useState, useRef, useEffect, type FormEvent } from "react";
import "./App.css";

type Engine = "vllm" | "sglang";

interface Message {
  role: "user" | "assistant";
  content: string;
  engine?: Engine;
}

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [engine, setEngine] = useState<Engine>("vllm");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage: Message = { role: "user", content: trimmed };
    const history = [...messages, userMessage];
    setMessages(history);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          history: messages.map(({ role, content }) => ({ role, content })),
          engine,
        }),
      });

      if (!res.ok || !res.body) throw new Error("Failed to fetch");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = "";

      setMessages([...history, { role: "assistant", content: "", engine }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        assistantContent += decoder.decode(value, { stream: true });
        setMessages([
          ...history,
          { role: "assistant", content: assistantContent, engine },
        ]);
      }
    } catch {
      setMessages([
        ...history,
        { role: "assistant", content: "Error: Failed to get response." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>LLM Chatbot</h1>
        <div className="engine-toggle">
          {(["vllm", "sglang"] as Engine[]).map((e) => (
            <button
              key={e}
              type="button"
              className={engine === e ? "active" : ""}
              onClick={() => setEngine(e)}
              disabled={isLoading}
            >
              {e === "vllm" ? "vLLM" : "SGLang"}
            </button>
          ))}
        </div>
      </header>
      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="empty-state">Send a message to start chatting.</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <span className="message-role">
              {msg.role === "user"
                ? "You"
                : msg.engine === "sglang"
                  ? "AI · SGLang"
                  : "AI · vLLM"}
            </span>
            <p className="message-content">{msg.content}</p>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

export default App;
