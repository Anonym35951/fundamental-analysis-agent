import { Component, type ErrorInfo, type ReactNode } from "react";

type ErrorBoundaryProps = {
  children: ReactNode;
};

type ErrorBoundaryState = {
  error: Error | null;
};

export default class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Unhandled UI error caught by ErrorBoundary:", error, info);
  }

  handleReload = () => {
    this.setState({ error: null });
    window.location.reload();
  };

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "60vh",
            padding: "48px 24px",
            textAlign: "center",
            gap: "16px",
          }}
        >
          <h2 style={{ margin: 0, fontSize: "1.5rem" }}>
            Etwas ist schiefgelaufen.
          </h2>
          <p style={{ margin: 0, opacity: 0.8, maxWidth: "480px" }}>
            Diese Ansicht konnte nicht geladen werden. Bitte lade die Seite
            neu. Falls das Problem weiterhin besteht, kontaktiere den
            Support.
          </p>
          <button
            onClick={this.handleReload}
            style={{
              padding: "10px 20px",
              borderRadius: "10px",
              border: "1px solid rgba(148, 163, 184, 0.4)",
              background: "rgba(59, 130, 246, 0.15)",
              color: "inherit",
              cursor: "pointer",
              fontSize: "0.95rem",
            }}
          >
            Seite neu laden
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
