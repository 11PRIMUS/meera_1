import { useCallback, useEffect, useMemo, useState } from "react";
import { createClient } from "@supabase/supabase-js";
import "./App.css";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

const supabaseClient =
  supabaseUrl && supabaseAnonKey ? createClient(supabaseUrl, supabaseAnonKey) : null;

export default function App() {
  const [session, setSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [diaryEntries, setDiaryEntries] = useState([]);
  const [messageInput, setMessageInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [diaryOpen, setDiaryOpen] = useState(false);
  const [activeDiaryIndex, setActiveDiaryIndex] = useState(0);

  const isEnvReady = Boolean(supabaseClient && apiBaseUrl);
  const statusLabel = session ? "Online" : "Offline";
  const userId = session?.user?.id;

  const handleError = useCallback((err) => {
    console.error(err);
    setError(typeof err === "string" ? err : err?.message ?? "Something went wrong.");
  }, []);

  const loadHistory = useCallback(
    async (uid) => {
      try {
        if (!apiBaseUrl) throw new Error("API endpoint not configured.");
        const response = await fetch(`${apiBaseUrl}/api/chat/history/${uid}`);
        if (!response.ok) throw new Error("Unable to fetch chat history.");
        const data = await response.json();
        setMessages(Array.isArray(data) ? data : []);
      } catch (err) {
        handleError(err);
      }
    },
    [apiBaseUrl, handleError]
  );

  const loadDiary = useCallback(
    async (uid) => {
      try {
        if (!apiBaseUrl) throw new Error("API endpoint not configured.");
        const response = await fetch(`${apiBaseUrl}/api/diary/${uid}`);
        if (!response.ok) throw new Error("Unable to fetch diary.");
        const data = await response.json();
        setDiaryEntries(Array.isArray(data?.entries) ? data.entries : []);
      } catch (err) {
        handleError(err);
      }
    },
    [apiBaseUrl, handleError]
  );

  const refreshUserData = useCallback(
    async (uid) => Promise.all([loadHistory(uid), loadDiary(uid)]),
    [loadDiary, loadHistory]
  );

  useEffect(() => {
    if (!supabaseClient || !apiBaseUrl) return;

    supabaseClient.auth.getSession().then(({ data }) => {
      setSession(data.session);
      if (data.session?.user?.id) {
        refreshUserData(data.session.user.id);
      }
    });

    const { data } = supabaseClient.auth.onAuthStateChange((_event, newSession) => {
      setSession(newSession);
      setError("");
      if (newSession?.user?.id) {
        refreshUserData(newSession.user.id);
      } else {
        setMessages([]);
        setDiaryEntries([]);
        setDiaryOpen(false);
        setActiveDiaryIndex(0);
      }
    });

    return () => {
      data?.subscription?.unsubscribe();
    };
  }, [apiBaseUrl, refreshUserData, supabaseClient]);

  useEffect(() => {
    setActiveDiaryIndex(0);
  }, [diaryEntries.length]);

  const handleDiaryNavigate = useCallback(
    (direction) => {
      setActiveDiaryIndex((current) => {
        const next = current + direction;
        if (next < 0 || next >= diaryEntries.length) return current;
        return next;
      });
    },
    [diaryEntries.length]
  );

  const activeDiaryEntry = diaryEntries[activeDiaryIndex] ?? null;

  const handleLogin = async () => {
    if (!supabaseClient) return;
    setError("");
    await supabaseClient.auth.signInWithOAuth({
      provider: "google",
      options: { queryParams: { prompt: "consent" } },
    });
  };

  const handleSignOut = async () => {
    if (!supabaseClient) return;
    await supabaseClient.auth.signOut();
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!userId || !messageInput.trim()) return;

    setBusy(true);
    setError("");
    try {
      if (!apiBaseUrl) throw new Error("API endpoint not configured.");
      const response = await fetch(`${apiBaseUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, message: messageInput.trim() }),
      });
      if (!response.ok) throw new Error("Unable to send message.");
      const data = await response.json();
      setMessages(Array.isArray(data?.history) ? data.history : []);
      if (data?.diary_entry) {
        setDiaryEntries((current) => {
          const filtered = current.filter((entry) => entry.id !== data.diary_entry.id);
          return [data.diary_entry, ...filtered];
        });
      } else {
        await loadDiary(userId);
      }
      setMessageInput("");
    } catch (err) {
      handleError(err);
    } finally {
      setBusy(false);
    }
  };

  const diaryEmptyState = useMemo(
    () => (
      <div className="diary-empty">
        <p>Diary entries unlock once every 24 hours after you finish chatting.</p>
      </div>
    ),
    []
  );

  const diaryStatusLabel = useMemo(() => {
    if (!activeDiaryEntry) return "No reflections yet";
    const createdAt = activeDiaryEntry.created_at ? new Date(activeDiaryEntry.created_at) : null;
    if (!createdAt || Number.isNaN(createdAt.getTime())) return "Reflection scheduled soon";
    return `Daily log · ${createdAt.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    })}`;
  }, [activeDiaryEntry]);

  const diaryPageCounter = diaryEntries.length
    ? `Day ${diaryEntries.length - activeDiaryIndex} of ${diaryEntries.length}`
    : "No entries yet";

  if (!isEnvReady) {
    return (
      <main className="app-shell centered">
        <section className="auth-panel">
          <h2>Missing configuration</h2>
          <p>Add <code>.env</code> with VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY, and VITE_API_BASE_URL.</p>
        </section>
      </main>
    );
  }

  return (
    <>
      <div className="aurora" aria-hidden="true">
        <span className="orb orb-one" />
        <span className="orb orb-two" />
        <span className="orb orb-three" />
      </div>
      <main className="app-shell">
        <section className="hero">
          <p className="eyebrow">Presence · Memory · Care</p>
          <h1>Share your day and see how Meera remembers.</h1>
          <p className="hero-copy">
            A calmer companion that keeps a gentle diary of every conversation so you can revisit the small details
            that matter.
          </p>
          <div className="hero-stats">
            <article>
              <span className="stat-label">Diary streak</span>
              <strong>7 days</strong>
            </article>
            <article>
              <span className="stat-label">Moments saved</span>
              <strong>{Math.max(diaryEntries.length, 0)}</strong>
            </article>
            <article>
              <span className="stat-label">Mood average</span>
              <strong>Soft Glow</strong>
            </article>
          </div>
        </section>

        {!session && (
          <section className="auth-panel">
            <div className="auth-copy">
              <h2>Welcome back</h2>
              <p>Sign in with Google to start chatting and peek into Meera&apos;s diary notes about your sessions.</p>
            </div>
            <button type="button" onClick={handleLogin}>
              Continue with Google
            </button>
            <p className="auth-subcopy">Secure Supabase Auth · Private to you</p>
          </section>
        )}

        {session && (
          <section className="chat-panel">
            <header className="chat-header">
              <div>
                <p className="eyebrow">Now chatting</p>
                <h2>Meera</h2>
              </div>
              <div className="header-actions">
                <p className="status-pill">{statusLabel}</p>
                <button type="button" onClick={handleSignOut} className="ghost">
                  Sign out
                </button>
              </div>
            </header>

            {error && <div className="error-banner">{error}</div>}

            <div className="panes">
              <div className="chat-pane">
                <div className="pane-heading">
                  <h3>Conversation</h3>
                  <p>Meera saves one diary reflection per day after your conversation winds down.</p>
                </div>
                <ul id="messageList" aria-live="polite">
                  {messages.map((message) => (
                    <li key={message.id ?? `${message.created_at}-${message.role}`} className={`bubble ${message.role}`}>
                      {message.content}
                    </li>
                  ))}
                </ul>
                <form onSubmit={handleSubmit}>
                  <label className="sr-only" htmlFor="messageInput">
                    Message Meera
                  </label>
                  <textarea
                    id="messageInput"
                    placeholder="Share how your day is going..."
                    rows={3}
                    required
                    value={messageInput}
                    onChange={(event) => setMessageInput(event.target.value)}
                    disabled={busy}
                  />
                  <button type="submit" disabled={busy}>
                    {busy ? "Sending..." : "Send reflection"}
                  </button>
                </form>
              </div>

              <div className={`diary-box ${diaryOpen ? "is-open" : "is-collapsed"}`}>
                <button
                  type="button"
                  className="diary-trigger"
                  aria-expanded={diaryOpen}
                  aria-controls="diaryPanel"
                  onClick={() => setDiaryOpen((open) => !open)}
                >
                  <div className="diary-trigger-copy">
                    <p className="eyebrow">Meera&apos;s Diary</p>
                    <strong>{diaryStatusLabel}</strong>
                    <span className="diary-hint">
                      Entries are written once every 24 hours, usually after you wrap the day.
                    </span>
                  </div>
                  <span className="diary-trigger-action">{diaryOpen ? "Hide log" : "View log"}</span>
                </button>

                {diaryOpen && (
                  <aside className="diary-pane diary-panel" id="diaryPanel">
                    <div className="pane-heading">
                      <h3>Meera&apos;s Diary</h3>
                      <p className="diary-hint">Daily reflections appear here after your sessions.</p>
                    </div>
                    <div className="diary-page">
                      {activeDiaryEntry ? (
                        <article className="diary-entry diary-page-content" key={activeDiaryEntry.id ?? activeDiaryEntry.created_at}>
                          <h4>{activeDiaryEntry.title}</h4>
                          <p>{activeDiaryEntry.content}</p>
                        </article>
                      ) : (
                        diaryEmptyState
                      )}
                    </div>
                    <div className="diary-nav">
                      <button
                        type="button"
                        className="ghost"
                        onClick={() => handleDiaryNavigate(1)}
                        disabled={activeDiaryIndex >= diaryEntries.length - 1}
                      >
                        Earlier entry
                      </button>
                      <p className="diary-page-counter">{diaryPageCounter}</p>
                      <button
                        type="button"
                        className="ghost"
                        onClick={() => handleDiaryNavigate(-1)}
                        disabled={activeDiaryIndex <= 0}
                      >
                        Later entry
                      </button>
                    </div>
                  </aside>
                )}
              </div>
            </div>
          </section>
        )}
      </main>
    </>
  );
}
