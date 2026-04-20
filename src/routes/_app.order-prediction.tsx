import { createFileRoute } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { Sliders, Zap, Sun, Cloud, Loader2, MapPin, UtensilsCrossed } from "lucide-react";
import { predictKpt, fetchMerchants } from "../lib/api";

// City abbreviation → full name
const CITY_MAP: Record<string, string> = {
  Del: "Delhi", Mum: "Mumbai", Ban: "Bangalore", Hyd: "Hyderabad",
  Che: "Chennai", Kol: "Kolkata", Pun: "Pune", Jai: "Jaipur",
  Sur: "Surat", Ahm: "Ahmedabad",
};

// Tier abbreviation → readable tier
const TIER_MAP: Record<string, string> = {
  lar: "Large Chain", med: "Mid-size", sma: "Small", ind: "Independent",
};

function formatMerchantLabel(merchant: any): string {
  // Try to parse merchant_id like "Healthy_lar_Del_0"
  const parts = merchant.merchant_id?.split("_") ?? [];
  const num = parts[parts.length - 1];
  const cityAbbr = parts[parts.length - 2];
  const city = CITY_MAP[cityAbbr] ?? cityAbbr ?? merchant.city ?? "";
  const cuisine = merchant.cuisine ?? parts[0]?.replace(/_/g, " ") ?? "";
  return `${cuisine} Restaurant #${num} · ${city}`;
}

export const Route = createFileRoute("/_app/order-prediction")({
  component: OrderPredictionPage,
});

function OrderPredictionPage() {
  const [complexity, setComplexity] = useState(7);
  const [busyness, setBusyness] = useState(82);
  const [weather, setWeather] = useState<"clear" | "rain">("clear");
  
  const [merchants, setMerchants] = useState<any[]>([]);
  const [selectedMerchantId, setSelectedMerchantId] = useState("");
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchMerchants({ per_page: 100 })
      .then(data => {
        setMerchants(data);
        if (data.length > 0) setSelectedMerchantId(data[0].merchant_id);
      })
      .catch(err => console.error(err));
  }, []);

  const selectedMerchant = merchants.find(m => m.merchant_id === selectedMerchantId);

  const handlePredict = async () => {
    if (!selectedMerchant) return;
    setLoading(true);
    setError("");
    try {
      const now = new Date();
      const res = await predictKpt({
        merchant_id: selectedMerchant.merchant_id,
        cuisine: selectedMerchant.cuisine,
        order_complexity: complexity,
        hour: now.getHours(),
        day_of_week: now.getDay(),
        google_busyness_index: busyness,
        rain_flag: weather === "rain" ? 1 : 0,
        rain_severity: weather === "rain" ? 0.8 : 0,
      });
      setResult(res);
    } catch (err) {
      setError("Prediction failed. Make sure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-[1200px]">
      <div>
        <h1 className="text-3xl font-extrabold font-display">Order Prediction</h1>
        <p className="text-sm text-muted-foreground mt-1">Kitchen Preparation Time (KPT) manual forecasting engine.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-6">
        {/* Prediction Parameters */}
        <div className="bg-card rounded-xl p-6 space-y-6">
          <div className="flex items-center gap-2">
            <Sliders size={18} className="text-primary" />
            <h2 className="text-lg font-bold font-display">Prediction Parameters</h2>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase block mb-1.5">Merchant</label>
              <select
                value={selectedMerchantId}
                onChange={e => setSelectedMerchantId(e.target.value)}
                className="w-full bg-input rounded-lg py-3 px-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
              >
                {merchants.map(m => (
                  <option key={m.merchant_id} value={m.merchant_id}>
                    {formatMerchantLabel(m)}
                  </option>
                ))}
              </select>
            </div>

            {/* Selected merchant info card */}
            {selectedMerchant && (
              <div className="bg-input/60 rounded-lg px-4 py-3 flex items-center gap-4">
                <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                  <UtensilsCrossed size={18} className="text-primary" />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-bold text-foreground truncate">{selectedMerchant.cuisine} Kitchen</div>
                  <div className="flex items-center gap-3 mt-0.5 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <MapPin size={10} />
                      {CITY_MAP[selectedMerchant.merchant_id?.split("_").slice(-2, -1)[0]] ?? selectedMerchant.city}
                    </span>
                    <span className="capitalize">{TIER_MAP[selectedMerchant.merchant_id?.split("_")[1]] ?? selectedMerchant.merchant_tier}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                      selectedMerchant.has_iot_beacon ? "bg-teal/10 text-teal" : "bg-muted text-muted-foreground"
                    }`}>{selectedMerchant.has_iot_beacon ? "IoT Beacon" : "No IoT"}</span>
                  </div>
                </div>
                <div className="ml-auto text-right shrink-0">
                  <div className="text-xs text-muted-foreground">Bias</div>
                  <div className={`text-sm font-bold ${
                    Math.abs(selectedMerchant.bias_mean) > 2 ? "text-warning" : "text-success"
                  }`}>{selectedMerchant.bias_mean > 0 ? "+" : ""}{selectedMerchant.bias_mean?.toFixed(1)}m</div>
                </div>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase block mb-1.5">Order Time</label>
              <div className="bg-input rounded-lg py-3 px-4 text-sm flex items-center justify-between opacity-80">
                <span>{new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                <span className="text-muted-foreground text-xs">⏰</span>
              </div>
            </div>
            <div>
              <label className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase block mb-1.5">Current Weather</label>
              <div className="flex rounded-lg overflow-hidden border border-border">
                <button onClick={() => setWeather("clear")} className={`flex-1 flex items-center justify-center gap-1.5 py-3 text-sm font-medium transition-colors ${weather === "clear" ? "bg-card text-primary" : "bg-input text-muted-foreground"}`}>
                  <Sun size={14} /> Clear
                </button>
                <button onClick={() => setWeather("rain")} className={`flex-1 flex items-center justify-center gap-1.5 py-3 text-sm font-medium transition-colors ${weather === "rain" ? "bg-card text-primary" : "bg-input text-muted-foreground"}`}>
                  <Cloud size={14} /> Rain
                </button>
              </div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">Order Complexity</label>
              <span className="text-sm font-bold text-primary">Level {String(complexity).padStart(2, "0")}</span>
            </div>
            <input
              type="range" min={1} max={10} value={complexity}
              onChange={(e) => setComplexity(Number(e.target.value))}
              className="w-full accent-primary h-1.5"
            />
            <div className="flex justify-between text-[10px] text-muted-foreground mt-1 font-bold tracking-wider uppercase">
              <span>Simple (Single Item)</span>
              <span>Complex (Bulk/Special)</span>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">Kitchen Busyness</label>
              <span className="text-sm font-bold text-primary">{busyness}%</span>
            </div>
            <input
              type="range" min={0} max={100} value={busyness}
              onChange={(e) => setBusyness(Number(e.target.value))}
              className="w-full accent-primary h-1.5"
            />
            <div className="flex justify-between text-[10px] text-muted-foreground mt-1 font-bold tracking-wider uppercase">
              <span>Quiet</span>
              <span>Peak Load</span>
            </div>
          </div>

          <button 
            onClick={handlePredict}
            disabled={loading || !selectedMerchant}
            className="w-full bg-primary text-primary-foreground font-bold text-base py-4 rounded-lg flex items-center justify-center gap-2 hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {loading ? <Loader2 size={18} className="animate-spin" /> : <Zap size={18} />} 
            {loading ? "Running SIREN Model..." : "Run Prediction Model"}
          </button>
          
          {error && <div className="text-primary text-sm text-center">{error}</div>}
        </div>

        {/* Results Panel */}
        <div className="space-y-4">
          {result && (
            <>
              <div className="bg-card rounded-xl p-6 text-center">
                <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">Predicted KPT</span>
                <div className="mt-2">
                  <span className="text-6xl font-extrabold font-display">{result.predicted_kpt_min.toFixed(1)}</span>
                  <span className="text-lg font-bold text-muted-foreground ml-1">MINS</span>
                </div>
                <div className="flex items-center justify-center gap-1.5 mt-2 text-xs text-teal font-medium">
                  <span className="h-2 w-2 rounded-full bg-teal" />
                  Confidence Interval: {result.confidence_interval[0].toFixed(1)} - {result.confidence_interval[1].toFixed(1)} mins
                </div>

                {result.for_signal_warning && (
                  <div className="mt-4 bg-warning/10 rounded-lg p-3 text-left">
                    <div className="flex items-center gap-1.5 text-sm font-bold text-warning">
                      <span>⚠</span> High Bias Warning
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      This merchant exhibits significant historical bias in their 'Food Ready' markings.
                    </p>
                  </div>
                )}
              </div>

              <div className="bg-card rounded-xl p-6">
                <h3 className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase mb-4">Feature Contributions</h3>
                {result.feature_contributions.map((item: any) => (
                  <div key={item.feature} className="mb-4">
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="font-medium">{item.feature.replace(/_/g, ' ')}</span>
                      <span className="font-bold text-primary">{(item.importance * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-accent rounded-full overflow-hidden">
                      <div className="h-full bg-primary rounded-full" style={{ width: `${item.importance * 100}%` }} />
                    </div>
                  </div>
                ))}
                <div className="flex items-center justify-between mt-4 pt-4 border-t border-border/50 text-xs text-muted-foreground">
                  <span>Model: SIREN_XGBoost</span>
                  <span className="text-primary font-medium">Recommended offset: {result.recommended_dispatch_offset_min}m</span>
                </div>
              </div>
            </>
          )}
          {!result && !loading && (
             <div className="bg-card rounded-xl p-6 text-center text-muted-foreground h-full flex flex-col items-center justify-center min-h-[300px]">
               <Sliders size={32} className="mb-4 opacity-50" />
               <p>Configure parameters and run the model to view predictions.</p>
             </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Lock({ size, className }: { size: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className={className}>
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}
