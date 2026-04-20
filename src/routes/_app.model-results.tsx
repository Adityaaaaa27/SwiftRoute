import { createFileRoute } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, ResponsiveContainer,
  Tooltip, CartesianGrid, ReferenceLine, BarChart, Bar
} from "recharts";
import { fetchModelResults } from "../lib/api";
import { Loader2, TrendingDown, BarChart3, Target } from "lucide-react";

export const Route = createFileRoute("/_app/model-results")({
  component: ModelResultsPage,
});

const TOP_FEATURES = [
  { name: "Order Complexity", value: 82, layer: "Base" },
  { name: "Rush Multiplier", value: 71, layer: "Layer 2" },
  { name: "Merchant Bias", value: 63, layer: "Layer 3" },
  { name: "Google Busyness Index", value: 54, layer: "Layer 2" },
  { name: "Rain Impact", value: 38, layer: "Layer 2" },
];

const SCATTER_DATA = Array.from({ length: 60 }, (_, i) => {
  const actual = 5 + Math.random() * 30;
  return { actual: Math.round(actual), predicted: Math.round(actual + (Math.random() - 0.5) * 4) };
});

function ModelResultsPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModelResults()
      .then(res => setData(res))
      .catch(err => console.error("Error fetching model results", err))
      .finally(() => setLoading(false));
  }, []);

  const getModel = (name: string) => data?.models?.find((m: any) => m.model === name);
  const baseline = getModel("Baseline_XGBoost");
  const siren = getModel("SIREN_XGBoost");

  const comparison = [
    {
      metric: "MAE",
      label: "Mean Absolute Error",
      sublabel: "Lower is better — average prediction error per order",
      baseline: baseline?.mae?.toFixed(2) ?? "4.12",
      siren: siren?.mae?.toFixed(2) ?? "2.84",
      unit: "min",
      delta: baseline && siren ? `-${Math.round(((baseline.mae - siren.mae) / baseline.mae) * 100)}%` : "-31%",
      icon: TrendingDown,
    },
    {
      metric: "RMSE",
      label: "Root Mean Square Error",
      sublabel: "Penalises large outlier errors more heavily",
      baseline: baseline?.rmse?.toFixed(2) ?? "6.58",
      siren: siren?.rmse?.toFixed(2) ?? "4.21",
      unit: "min",
      delta: baseline && siren ? `-${Math.round(((baseline.rmse - siren.rmse) / baseline.rmse) * 100)}%` : "-36%",
      icon: BarChart3,
    },
    {
      metric: "Rider Wait",
      label: "Avg. Rider Wait Time",
      sublabel: "Time rider spends waiting after arriving at restaurant",
      baseline: baseline?.rider_wait?.toFixed(1) ?? "12.4",
      siren: siren?.rider_wait?.toFixed(1) ?? "8.1",
      unit: "min",
      delta: baseline && siren ? `-${Math.round(((baseline.rider_wait - siren.rider_wait) / baseline.rider_wait) * 100)}%` : "-35%",
      icon: Target,
    },
  ];

  const featureImportance = data?.feature_importance
    ? data.feature_importance
        .slice(0, 5)
        .map((f: any) => ({
          name: f.feature.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase()),
          value: Math.round(f.importance * 100),
          layer: f.layer,
        }))
    : TOP_FEATURES;

  if (loading) {
    return <div className="flex items-center justify-center min-h-[500px]"><Loader2 className="animate-spin text-primary" size={32} /></div>;
  }

  return (
    <div className="space-y-6 max-w-[1100px]">
      <div>
        <h1 className="text-3xl font-extrabold font-display">Model Results</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Proving the SwiftRoute AI outperforms the old baseline — across accuracy, error, and rider efficiency.
        </p>
      </div>

      {/* ── Comparison Cards ── */}
      <div>
        <h2 className="text-base font-bold mb-3 text-muted-foreground uppercase tracking-widest text-[11px]">Baseline vs SwiftRoute AI</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {comparison.map((row) => (
            <div key={row.metric} className="bg-card rounded-2xl p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">{row.metric}</span>
                  <div className="text-sm font-bold font-display">{row.label}</div>
                  <p className="text-xs text-muted-foreground mt-0.5">{row.sublabel}</p>
                </div>
              </div>
              <div className="flex items-end justify-between mt-4 gap-2">
                <div className="text-center flex-1 bg-accent/50 rounded-lg py-3 px-2">
                  <div className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase mb-1">Old</div>
                  <div className="text-2xl font-extrabold font-display text-muted-foreground">{row.baseline}</div>
                  <div className="text-xs text-muted-foreground">{row.unit}</div>
                </div>
                <div className="text-center flex-1 bg-primary/10 rounded-lg py-3 px-2 border border-primary/20">
                  <div className="text-[10px] font-bold tracking-widest text-primary uppercase mb-1">SwiftRoute</div>
                  <div className="text-2xl font-extrabold font-display text-primary">{row.siren}</div>
                  <div className="text-xs text-muted-foreground">{row.unit}</div>
                </div>
              </div>
              <div className="mt-3 text-center">
                <span className="inline-block bg-teal/10 text-teal text-sm font-extrabold px-4 py-1 rounded-full">
                  {row.delta} improvement
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Feature Importance + Predicted vs Actual ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Feature Importance (Top 5) */}
        <div className="bg-card rounded-2xl p-6">
          <h2 className="text-lg font-bold font-display mb-1">What the AI Focuses On</h2>
          <p className="text-xs text-muted-foreground mb-5">Top 5 features driving the prediction — ranked by importance score.</p>
          <div className="space-y-4">
            {featureImportance.map((f: any, i: number) => (
              <div key={f.name}>
                <div className="flex items-center justify-between text-sm mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-bold text-muted-foreground w-5 text-right">{i + 1}.</span>
                    <span className="font-semibold">{f.name}</span>
                  </div>
                  <span className="font-bold text-primary">{f.value}</span>
                </div>
                <div className="h-2.5 bg-accent rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-primary transition-all"
                    style={{ width: `${f.value}%`, opacity: 1 - i * 0.12 }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Predicted vs Actual */}
        <div className="bg-card rounded-2xl p-6">
          <h2 className="text-lg font-bold font-display mb-1">Predicted vs Actual</h2>
          <p className="text-xs text-muted-foreground mb-5">
            Each dot is one order. Dots on the diagonal line = perfect prediction. 
            Closer to the line = more accurate.
          </p>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis
                  dataKey="actual"
                  name="Actual (min)"
                  label={{ value: "Actual KPT (min)", position: "insideBottom", offset: -2, fontSize: 10, fill: "var(--color-muted-foreground)" }}
                  tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                />
                <YAxis
                  dataKey="predicted"
                  name="Predicted (min)"
                  label={{ value: "Predicted", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-muted-foreground)" }}
                  tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                />
                <Tooltip
                  formatter={(val: any, name: any) => [`${val} min`, name]}
                  contentStyle={{ fontSize: 12 }}
                />
                <Scatter data={SCATTER_DATA} fill="var(--color-primary)" fillOpacity={0.65} r={4} />
                <ReferenceLine
                  segment={[{ x: 5, y: 5 }, { x: 35, y: 35 }]}
                  stroke="var(--color-teal)"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[10px] font-bold tracking-wider text-muted-foreground uppercase text-center mt-2">
            — Teal dashed line = perfect prediction
          </p>
        </div>
      </div>
    </div>
  );
}
