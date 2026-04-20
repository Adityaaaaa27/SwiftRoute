import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { useAuth } from "../lib/auth-context";
import { Lock, AtSign, Shield, ChevronDown, ArrowRight, BarChart3 } from "lucide-react";

export const Route = createFileRoute("/")({
  component: LoginPage,
});

function LoginPage() {
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("");
  const [loading, setLoading] = useState(false);
  const [showCover, setShowCover] = useState(true);
  const [coverLeaving, setCoverLeaving] = useState(false);

  if (isAuthenticated) {
    navigate({ to: "/dashboard" });
    return null;
  }

  const handleEnter = () => {
    setCoverLeaving(true);
    setTimeout(() => setShowCover(false), 600);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    await login(email, password, role || "admin");
    setLoading(false);
    navigate({ to: "/dashboard" });
  };

  return (
    <div className="relative min-h-screen bg-background overflow-hidden">
      {/* ── Cover / Splash Page ── */}
      {showCover && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 50,
            backgroundImage: "url('/swiftroute-cover.png')",
            backgroundSize: "cover",
            backgroundPosition: "center",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "flex-end",
            paddingBottom: "72px",
            animation: coverLeaving
              ? "coverSlideUp 0.6s cubic-bezier(0.4,0,0.2,1) forwards"
              : "coverFadeIn 0.8s ease forwards",
          }}
        >
          {/* subtle dark overlay at bottom for button visibility */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              background: "linear-gradient(to top, rgba(0,0,0,0.45) 0%, transparent 55%)",
              pointerEvents: "none",
            }}
          />

          <button
            id="cover-login-btn"
            onClick={handleEnter}
            style={{
              position: "relative",
              zIndex: 1,
              backgroundColor: "#ffffff",
              color: "#e53935",
              fontWeight: 800,
              fontSize: "1.05rem",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              padding: "16px 64px",
              borderRadius: "999px",
              border: "none",
              cursor: "pointer",
              boxShadow: "0 8px 32px rgba(0,0,0,0.25)",
              transition: "transform 0.15s ease, box-shadow 0.15s ease",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.transform = "scale(1.04)";
              (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 12px 40px rgba(0,0,0,0.35)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.transform = "scale(1)";
              (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 8px 32px rgba(0,0,0,0.25)";
            }}
          >
            Login
          </button>
        </div>
      )}

      {/* ── Normal Login Page ── */}
      <div
        className="flex min-h-screen bg-background"
        style={{
          opacity: showCover ? 0 : 1,
          transition: "opacity 0.4s ease",
        }}
      >
        {/* Left panel */}
        <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12 bg-gradient-to-b from-siren-light to-background relative">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-3xl font-extrabold text-primary font-display">SwiftRoute</span>
              <div className="h-6 w-px bg-muted-foreground/30 mx-1" />
              <div className="text-xs leading-tight text-muted-foreground">
                <div>Operational</div>
                <div>Intel</div>
              </div>
            </div>
          </div>

          <div>
            <h1 className="text-5xl font-extrabold leading-tight font-display text-foreground">
              Command the <span className="text-primary">Pulse</span>
              <br />of Logistics.
            </h1>
            <p className="mt-6 text-base text-muted-foreground max-w-md leading-relaxed">
              Access real-time order predictions, merchant signals, and simulation tools in one clinical interface.
            </p>
          </div>

          <div className="flex items-center gap-3 bg-card/80 backdrop-blur-sm rounded-xl px-4 py-3 w-fit">
            <div className="h-10 w-10 rounded-lg bg-success/10 flex items-center justify-center">
              <BarChart3 size={20} className="text-success" />
            </div>
            <div>
              <div className="text-[10px] font-semibold tracking-widest text-muted-foreground uppercase">System Status</div>
              <div className="text-sm font-bold text-foreground">99.9% Signal Quality</div>
            </div>
          </div>
        </div>

        {/* Right panel — Login form */}
        <div className="flex-1 flex items-center justify-center p-8">
          <form onSubmit={handleSubmit} className="w-full max-w-md space-y-6">
            <div>
              <h2 className="text-3xl font-extrabold font-display text-foreground">Welcome Back</h2>
              <p className="mt-1 text-sm text-muted-foreground">Log in to your operational dashboard</p>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase block mb-1.5">Access Level</label>
                <div className="relative">
                  <Shield size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <select
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="w-full bg-input rounded-lg py-3 pl-10 pr-10 text-sm font-medium text-foreground appearance-none focus:outline-none focus:ring-2 focus:ring-primary/30"
                  >
                    <option value="">Select your role</option>
                    <option value="admin">Admin</option>
                    <option value="ops_manager">Operations Manager</option>
                  </select>
                  <ChevronDown size={16} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" />
                </div>
              </div>

              <div>
                <label className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase block mb-1.5">Email Address</label>
                <div className="relative">
                  <AtSign size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="name@zomato.com"
                    className="w-full bg-input rounded-lg py-3 pl-10 pr-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">Password</label>
                  <button type="button" className="text-[10px] font-bold tracking-wide text-primary uppercase">Forgot?</button>
                </div>
                <div className="relative">
                  <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full bg-input rounded-lg py-3 pl-10 pr-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                  />
                </div>
              </div>

              <label className="flex items-center gap-2 cursor-pointer">
                <div className="h-4 w-4 rounded border border-border bg-card" />
                <span className="text-sm text-muted-foreground">Remember this session</span>
              </label>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-primary text-primary-foreground font-bold text-base py-4 rounded-lg flex items-center justify-center gap-2 hover:opacity-90 transition-opacity"
            >
              Sign In <ArrowRight size={18} />
            </button>

            <div className="text-center space-y-2 pt-4">
              <p className="text-xs text-muted-foreground">Secured by <span className="font-bold text-foreground">SwiftRoute Core Identity</span></p>
              <div className="flex items-center justify-center gap-4 text-xs text-primary font-medium">
                <span>Security Policy</span>
                <span>Internal Support</span>
              </div>
            </div>
          </form>
        </div>
      </div>

      {/* ── Keyframe Animations ── */}
      <style>{`
        @keyframes coverFadeIn {
          from { opacity: 0; transform: scale(1.03); }
          to   { opacity: 1; transform: scale(1); }
        }
        @keyframes coverSlideUp {
          from { opacity: 1; transform: translateY(0) scale(1); }
          to   { opacity: 0; transform: translateY(-60px) scale(0.97); }
        }
      `}</style>
    </div>
  );
}
