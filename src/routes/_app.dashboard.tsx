import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AlertTriangle, TrendingUp, TrendingDown, CheckCircle, AlertCircle } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { dashboardSummary as mockSummary, kptTrendData, liveOrders as mockLiveOrders } from "../lib/mock-data";
import { fetchDashboardSummary, fetchLiveOrders } from "../lib/api";

export const Route = createFileRoute("/_app/dashboard")({
  component: DashboardPage,
});

function MetricCard({ title, value, unit, change, label, status, changeType }: {
  title: string; value: string | number; unit?: string; change?: string; label?: string; status?: string; changeType?: "up" | "down" | "neutral";
}) {
  return (
    <div className="bg-card rounded-xl p-5 flex flex-col gap-2">
      <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">{title}</span>
      <div className="flex items-baseline gap-1.5">
        <span className="text-3xl font-extrabold font-display text-foreground">{value}</span>
        {unit && <span className="text-sm text-muted-foreground font-medium">{unit}</span>}
      </div>
      {change && (
        <div className="flex items-center gap-1 text-xs font-medium">
          {changeType === "up" ? <TrendingUp size={12} className="text-primary" /> : changeType === "down" ? <TrendingDown size={12} className="text-success" /> : null}
          <span className={changeType === "up" ? "text-primary" : "text-success"}>{change}</span>
          {label && <span className="text-muted-foreground">{label}</span>}
        </div>
      )}
      {status && (
        <div className="flex items-center gap-1 text-xs font-medium">
          {status === "Within Limit" || status === "Optimal" ? <CheckCircle size={12} className="text-teal" /> : <AlertCircle size={12} className="text-primary" />}
          <span className={status === "Within Limit" || status === "Optimal" ? "text-teal" : "text-primary"}>{status}</span>
        </div>
      )}
    </div>
  );
}

function DashboardPage() {
  const [summary, setSummary] = useState<any>(null);
  const [orders, setOrders] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [sumData, ordData] = await Promise.all([
          fetchDashboardSummary(),
          fetchLiveOrders(10)
        ]);
        setSummary(sumData);
        setOrders(ordData);
      } catch (err) {
        console.error("Failed to fetch dashboard data", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const s = summary ? {
    averageKpt: { value: summary.avg_kpt_today, unit: "min", change: "+0%", label: "vs LW" },
    riderWaitTime: { value: summary.rider_wait_avg, unit: "min", change: "-0%", label: "vs LW" },
    p50EtaError: { value: summary.p50_eta_error, unit: "min", status: "Within Limit" },
    p90EtaError: { value: summary.p90_eta_error, unit: "min", status: summary.p90_eta_error > 10 ? "High Variance" : "Optimal" },
    totalOrders: { value: summary.total_orders_today, change: "+0%", label: "Today" },
    onTimeRate: { value: (summary.on_time_rate * 100).toFixed(1), unit: "%", status: "Optimal" },
  } : mockSummary;

  const displayOrders = orders.length > 0 ? orders.map(o => ({
    id: o.order_id,
    merchant: o.merchant_name,
    cuisine: o.cuisine,
    predKpt: `${o.predicted_kpt} MIN`,
    status: o.status.toUpperCase(),
    rider: o.rider_assigned ? "Assigned" : "Searching",
    riderDist: "",
    signal: o.for_signal_quality === "excellent" || o.for_signal_quality === "good" ? "clean" : "suspicious"
  })) : mockLiveOrders;

  const rushIndex = summary && Object.keys(summary.city_rush_index).length > 0 
    ? (Object.values(summary.city_rush_index)[0] as number / 10).toFixed(1)
    : "8.4";

  return (
    <div className="space-y-6 max-w-[1200px]">
      {/* Alert Banner */}
      <div className="bg-siren-light rounded-xl px-5 py-4 flex items-start gap-3">
        <AlertTriangle size={20} className="text-primary mt-0.5 shrink-0" />
        <div className="flex-1">
          <span className="text-xs font-bold tracking-wider text-primary uppercase">System Alert: </span>
          <span className="text-sm text-foreground">Abnormal KPT spike detected in BKC/Andheri sector (+12.4%). Recommendation: Adjust Rider Buffer.</span>
        </div>
        <button className="text-xs font-bold tracking-wider text-primary uppercase shrink-0">Acknowledge</button>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard title="Average KPT" value={s.averageKpt.value} unit={s.averageKpt.unit} change={s.averageKpt.change} label={s.averageKpt.label} changeType="up" />
        <MetricCard title="Rider Wait Time" value={s.riderWaitTime.value} unit={s.riderWaitTime.unit} change={s.riderWaitTime.change} label={s.riderWaitTime.label} changeType="down" />
        <MetricCard title="P50 ETA Error" value={`±${s.p50EtaError.value}`} unit={s.p50EtaError.unit} status={s.p50EtaError.status} />
        <MetricCard title="P90 ETA Error" value={s.p90EtaError.value} unit={s.p90EtaError.unit} status={s.p90EtaError.status} />
        <MetricCard title="Total Orders" value={s.totalOrders.value} change={s.totalOrders.change} label={s.totalOrders.label} changeType="up" />
        <MetricCard title="On-Time Rate" value={s.onTimeRate.value} unit={s.onTimeRate.unit} status={s.onTimeRate.status} />
      </div>

      {/* Chart Row */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-4">
        {/* KPT Trend Chart */}
        <div className="bg-card rounded-xl p-6">
          <div className="flex items-center justify-between mb-1">
            <div>
              <h2 className="text-lg font-bold font-display">KPT Trend Analysis</h2>
              <p className="text-xs text-muted-foreground">Real-time Kitchen Prep Time Performance</p>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-primary" /> Actual</span>
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-muted-foreground/40" /> Predicted</span>
            </div>
          </div>
          <div className="h-[280px] mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={kptTrendData} barGap={4}>
                <XAxis dataKey="time" tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} />
                <YAxis hide />
                <Tooltip />
                <Bar dataKey="predicted" fill="var(--color-muted)" radius={[4, 4, 0, 0]} barSize={36} />
                <Bar dataKey="actual" fill="var(--color-primary)" radius={[4, 4, 0, 0]} barSize={36} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Rush Index */}
        <div className="bg-primary rounded-xl p-6 text-primary-foreground flex flex-col justify-between">
          <div>
            <h3 className="text-lg font-bold font-display">Rush Index</h3>
            <p className="text-xs opacity-75">Market Momentum Score</p>
          </div>
          <div>
            <span className="text-7xl font-extrabold font-display">{rushIndex}</span>
            <div className="text-xs font-bold tracking-widest uppercase mt-2">Severe Demand</div>
          </div>
          <div className="bg-primary-foreground/10 rounded-lg px-4 py-3 flex items-center justify-between">
            <span className="text-sm font-medium">Supply Gap</span>
            <span className="text-sm font-bold">+18%</span>
          </div>
        </div>
      </div>

      {/* Live Orders */}
      <div className="bg-card rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold font-display">Live Operational Feed</h2>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
            <span className="text-xs font-bold tracking-wider text-primary uppercase">Live Updates</span>
          </div>
        </div>
        {loading ? (
           <div className="py-8 text-center text-muted-foreground">Loading live orders...</div>
        ) : (
        <table className="w-full">
          <thead>
            <tr className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">
              <th className="text-left pb-3">Order ID</th>
              <th className="text-left pb-3">Merchant</th>
              <th className="text-left pb-3">Cuisine</th>
              <th className="text-left pb-3">Pred. KPT</th>
              <th className="text-left pb-3">Status</th>
              <th className="text-left pb-3">Rider</th>
              <th className="text-left pb-3">Signal</th>
            </tr>
          </thead>
          <tbody>
            {displayOrders.map((order: any) => (
              <tr key={order.id} className="border-t border-border/50">
                <td className="py-4 text-sm font-bold font-display">{order.id}</td>
                <td className="py-4">
                  <div className="flex items-center gap-2">
                    <div className="h-8 w-8 rounded-full bg-accent flex items-center justify-center text-xs font-bold">🍕</div>
                    <span className="text-sm font-medium">{order.merchant}</span>
                  </div>
                </td>
                <td className="py-4 text-sm text-muted-foreground">{order.cuisine}</td>
                <td className="py-4">
                  <span className="bg-accent text-foreground text-xs font-bold px-2.5 py-1 rounded">{order.predKpt}</span>
                </td>
                <td className="py-4">
                  <span className={`text-[10px] font-bold tracking-wider uppercase px-2 py-1 rounded ${
                    order.status === "PREPARING" ? "bg-primary text-primary-foreground" :
                    order.status === "ASSIGNED" || order.status === "DELIVERED" ? "bg-teal text-teal-foreground" : "bg-accent text-foreground"
                  }`}>{order.status}</span>
                </td>
                <td className="py-4">
                  <span className="text-sm font-medium">{order.rider}</span>
                  <span className="text-xs text-muted-foreground ml-1">{order.riderDist}</span>
                </td>
                <td className="py-4">
                  <span className={`text-xs ${order.signal === "clean" ? "text-teal" : "text-primary"}`}>
                    {order.signal === "clean" ? "✓" : "⚠"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        )}
        <div className="mt-4 flex justify-center">
          <button className="text-xs font-bold tracking-widest uppercase text-muted-foreground border border-border rounded-lg px-6 py-2.5 hover:bg-accent transition-colors">
            View Full Operational Log
          </button>
        </div>
      </div>
    </div>
  );
}

