import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Save, Mail, MessageSquare, Webhook } from "lucide-react";
import { toast } from "sonner";

export const Route = createFileRoute("/_app/settings")({
  component: SettingsPage,
});

function SettingsPage() {
  const [theta, setTheta] = useState(0.84);
  const [layer1, setLayer1] = useState(true);
  const [layer2, setLayer2] = useState(true);
  const [layer3, setLayer3] = useState(false);
  const [dispatchOffset, setDispatchOffset] = useState(4);
  const [emailAlerts, setEmailAlerts] = useState(true);
  const [smsAlerts, setSmsAlerts] = useState(false);
  const [slackAlerts, setSlackAlerts] = useState(true);

  const handleSave = () => {
    toast.success("Configuration Staged", { description: "Threshold update ready for deployment" });
  };

  return (
    <div className="space-y-6 max-w-[1200px]">
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-6">
        {/* Left column */}
        <div className="space-y-6">
          {/* Theta Threshold */}
          <div className="bg-card rounded-xl p-6">
            <div className="flex items-center justify-between mb-1">
              <h2 className="text-2xl font-extrabold font-display">Theta Threshold</h2>
              <span className="bg-teal text-teal-foreground text-[10px] font-bold tracking-wider px-3 py-1 rounded-full uppercase">Live System</span>
            </div>
            <p className="text-sm text-muted-foreground mb-6">Configure the predictive confidence floor for automated routing.</p>

            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">Confidence Level</span>
              <span className="text-4xl font-extrabold font-display text-primary">{theta.toFixed(2)}</span>
            </div>
            <input
              type="range" min={0.1} max={1.0} step={0.01} value={theta}
              onChange={(e) => setTheta(Number(e.target.value))}
              className="w-full accent-primary h-2"
            />
            <div className="flex justify-between text-[10px] font-bold tracking-widest text-muted-foreground uppercase mt-1">
              <span>Conservative (0.1)</span>
              <span>Aggressive (1.0)</span>
            </div>

            <div className="grid grid-cols-3 gap-3 mt-6">
              {[
                { label: "AUTO-DISPATCH RATE", value: "+12.4%", color: "teal" },
                { label: "ORDER ACCURACY", value: "98.2%", color: "primary" },
                { label: "MANUAL OVERRIDES", value: "~42/hr", color: "destructive" },
              ].map((stat) => (
                <div key={stat.label} className="bg-accent rounded-lg p-3">
                  <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">{stat.label}</span>
                  <div className="text-xl font-extrabold font-display mt-1">{stat.value}</div>
                  <div className={`h-1 rounded-full mt-2 ${
                    stat.color === "teal" ? "bg-teal" : stat.color === "primary" ? "bg-primary" : "bg-destructive"
                  }`} />
                </div>
              ))}
            </div>
          </div>

          {/* Dispatch Optimization */}
          <div className="bg-card rounded-xl p-6">
            <h2 className="text-2xl font-extrabold font-display mb-1">Dispatch Optimization</h2>
            <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">Rider Dispatch Offset (Minutes)</span>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
              <div className="bg-accent rounded-lg p-4 flex items-center gap-3">
                <span className="text-lg">⏱</span>
                <input
                  type="number" value={dispatchOffset} min={0} max={10}
                  onChange={(e) => setDispatchOffset(Number(e.target.value))}
                  className="bg-transparent text-2xl font-extrabold font-display w-16 focus:outline-none"
                />
                <span className="text-sm text-muted-foreground">MIN</span>
              </div>
              <div className="bg-siren-light rounded-lg p-4 text-sm border-l-2 border-primary">
                <p className="italic text-muted-foreground">
                  "Setting this to <span className="text-primary font-bold">{dispatchOffset} min</span> currently matches your zone's median 'Arrival-to-Pickup' window."
                </p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-3">Buffer time before predicted order ready time to initiate rider search.</p>
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-6">
          {/* Operational Layers */}
          <div className="bg-card rounded-xl p-6">
            <h3 className="text-lg font-bold font-display mb-4">Operational Layers</h3>
            {[
              { label: "Layer 1: Fleet Base", sub: "Core algorithmic dispatching", checked: layer1, set: setLayer1 },
              { label: "Layer 2: Surge Mode", sub: "Peak demand velocity scaling", checked: layer2, set: setLayer2 },
              { label: "Layer 3: ML Refinement", sub: "Real-time localized learning", checked: layer3, set: setLayer3 },
            ].map((layer) => (
              <div key={layer.label} className="flex items-center justify-between py-3 border-b border-border/30 last:border-0">
                <div>
                  <div className="text-sm font-bold">{layer.label}</div>
                  <div className="text-xs text-muted-foreground">{layer.sub}</div>
                </div>
                <button
                  onClick={() => layer.set(!layer.checked)}
                  className={`w-11 h-6 rounded-full transition-colors relative ${layer.checked ? "bg-primary" : "bg-muted"}`}
                >
                  <span className={`absolute top-1 h-4 w-4 rounded-full bg-card shadow transition-transform ${layer.checked ? "left-6" : "left-1"}`} />
                </button>
              </div>
            ))}
          </div>

          {/* Alert Channels */}
          <div className="bg-card rounded-xl p-6">
            <h3 className="text-lg font-bold font-display mb-4">Alert Channels</h3>
            {[
              { icon: Mail, label: "Email Alerts", sub: "Daily operational summary", checked: emailAlerts, set: setEmailAlerts },
              { icon: MessageSquare, label: "SMS Notifications", sub: "Critical system failures only", checked: smsAlerts, set: setSmsAlerts },
              { icon: Webhook, label: "Slack Webhook", sub: "Real-time signal health", checked: slackAlerts, set: setSlackAlerts },
            ].map((channel) => (
              <div key={channel.label} className="flex items-center gap-3 py-3 border-b border-border/30 last:border-0">
                <div className="h-9 w-9 rounded-lg bg-accent flex items-center justify-center">
                  <channel.icon size={16} className="text-muted-foreground" />
                </div>
                <div className="flex-1">
                  <div className="text-sm font-bold">{channel.label}</div>
                  <div className="text-xs text-muted-foreground">{channel.sub}</div>
                </div>
                <input
                  type="checkbox" checked={channel.checked}
                  onChange={() => channel.set(!channel.checked)}
                  className="h-4 w-4 accent-primary rounded"
                />
              </div>
            ))}
          </div>

          <button onClick={handleSave} className="w-full bg-primary text-primary-foreground font-bold text-sm py-3.5 rounded-lg flex items-center justify-center gap-2 hover:opacity-90 transition-opacity">
            <Save size={16} /> SAVE CHANGES
          </button>
          <button className="w-full border border-border font-bold text-sm py-3.5 rounded-lg text-foreground hover:bg-accent transition-colors">
            Discard All Edits
          </button>
        </div>
      </div>

      {/* Audit Log */}
      <div className="bg-card rounded-xl p-6 text-center">
        <h3 className="text-lg font-bold font-display">Audit Log</h3>
        <p className="text-sm text-muted-foreground mt-1">Last change: Threshold adjusted from 0.82 to 0.84 by Admin on Oct 24, 14:02 PM.</p>
        <button className="text-xs font-bold tracking-widest text-primary uppercase mt-3">View Full Logs</button>
      </div>
    </div>
  );
}
