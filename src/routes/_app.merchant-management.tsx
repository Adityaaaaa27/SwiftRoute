import { createFileRoute } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { Download, Search, X, Flag, Loader2 } from "lucide-react";
import { BarChart, Bar, XAxis, ResponsiveContainer, LineChart, Line, Tooltip } from "recharts";
import { fetchMerchants, fetchMerchantDetail } from "../lib/api";

export const Route = createFileRoute("/_app/merchant-management")({
  component: MerchantManagementPage,
});

function MerchantManagementPage() {
  const [merchants, setMerchants] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detailData, setDetailData] = useState<any>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  
  const [page, setPage] = useState(1);
  const [totalEntities, setTotalEntities] = useState(1000); // Mock total for now
  
  const [cityFilter, setCityFilter] = useState("all");
  const [cuisineFilter, setCuisineFilter] = useState("all");
  const [tierFilter, setTierFilter] = useState("all");

  useEffect(() => {
    setLoading(true);
    fetchMerchants({ page, per_page: 20, city: cityFilter, cuisine: cuisineFilter, tier: tierFilter })
      .then(data => {
        setMerchants(data);
        if (data.length > 0 && !selectedId) {
          handleSelect(data[0].merchant_id);
        }
      })
      .catch(err => console.error("Error fetching merchants", err))
      .finally(() => setLoading(false));
  }, [page, cityFilter, cuisineFilter, tierFilter]);

  const handleSelect = async (id: string) => {
    setSelectedId(id);
    setDetailLoading(true);
    try {
      const data = await fetchMerchantDetail(id);
      setDetailData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setDetailLoading(false);
    }
  };

  const biasTrendData = detailData?.bias_trend?.map((bias: number, i: number) => ({
    day: i + 1,
    bias: bias
  })) || [];

  const kptDistData = detailData?.kpt_distribution ? [
    { range: "Min", count: detailData.kpt_distribution.min },
    { range: "P25", count: detailData.kpt_distribution.p25 },
    { range: "Med", count: detailData.kpt_distribution.median },
    { range: "P75", count: detailData.kpt_distribution.p75 },
    { range: "Max", count: detailData.kpt_distribution.max },
  ] : [];

  return (
    <div className="space-y-6 max-w-[1200px]">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold font-display">Merchant Management</h1>
          <p className="text-sm text-muted-foreground mt-1">Real-time performance auditing and behavior classification</p>
        </div>
        <button 
          onClick={() => selectedId ? window.location.href=`http://localhost:8000/api/merchants/${selectedId}/export` : null}
          className="bg-primary text-primary-foreground font-bold text-sm px-5 py-2.5 rounded-lg flex items-center gap-2 hover:opacity-90 transition-opacity"
        >
          <Download size={16} /> Export Orders
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <select value={cuisineFilter} onChange={e => setCuisineFilter(e.target.value)} className="bg-input rounded-lg py-2.5 px-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 min-w-[140px]">
          <option value="all">All Cuisines</option>
          <option value="Indian">Indian</option>
          <option value="Biryani">Biryani</option>
          <option value="Chinese">Chinese</option>
          <option value="Pizza">Pizza</option>
        </select>
        <select value={tierFilter} onChange={e => setTierFilter(e.target.value)} className="bg-input rounded-lg py-2.5 px-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 min-w-[140px]">
          <option value="all">All Tiers</option>
          <option value="1">Tier 1</option>
          <option value="2">Tier 2</option>
          <option value="3">Tier 3</option>
        </select>
        <select value={cityFilter} onChange={e => setCityFilter(e.target.value)} className="bg-input rounded-lg py-2.5 px-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 min-w-[140px]">
          <option value="all">All Cities</option>
          <option value="Mumbai">Mumbai</option>
          <option value="Bangalore">Bangalore</option>
          <option value="Delhi">Delhi</option>
        </select>
        <div className="flex-1" />
        <div className="bg-primary text-primary-foreground rounded-xl px-5 py-3 text-center">
          <div className="text-[10px] font-bold tracking-widest uppercase">Total Entities</div>
          <div className="text-2xl font-extrabold font-display">{totalEntities}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6">
        {/* Table */}
        <div className="bg-card rounded-xl overflow-hidden flex flex-col min-h-[500px]">
          {loading ? (
             <div className="flex-1 flex items-center justify-center"><Loader2 size={32} className="animate-spin text-muted-foreground" /></div>
          ) : (
          <>
            <table className="w-full">
              <thead>
                <tr className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">
                  <th className="text-left p-4">ID</th>
                  <th className="text-left p-4">Merchant Name</th>
                  <th className="text-left p-4">Cuisine</th>
                  <th className="text-left p-4">Tier</th>
                  <th className="text-left p-4">City</th>
                  <th className="text-right p-4">Bias</th>
                </tr>
              </thead>
              <tbody>
                {merchants.map((m: any) => (
                  <tr
                    key={m.merchant_id}
                    onClick={() => handleSelect(m.merchant_id)}
                    className={`border-t border-border/30 cursor-pointer transition-colors hover:bg-accent/50 ${
                      selectedId === m.merchant_id ? "bg-accent/50" : ""
                    }`}
                  >
                    <td className="p-4 text-xs text-muted-foreground">{m.merchant_id}</td>
                    <td className="p-4 text-sm font-bold">{m.merchant_name}</td>
                    <td className="p-4 text-sm text-muted-foreground">{m.cuisine}</td>
                    <td className="p-4">
                      <span className={`text-[10px] font-bold tracking-wider px-2 py-0.5 rounded ${
                        m.merchant_tier === "1" ? "bg-teal text-teal-foreground" :
                        m.merchant_tier === "2" ? "bg-primary/20 text-primary" :
                        "bg-accent text-foreground"
                      }`}>TIER {m.merchant_tier}</span>
                    </td>
                    <td className="p-4 text-sm">{m.city}</td>
                    <td className="p-4 text-sm font-bold text-right text-primary">{m.bias_mean.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-auto p-4 flex items-center justify-between text-xs text-muted-foreground border-t border-border/30">
              <span>Page {page}</span>
              <div className="flex items-center gap-1">
                <button disabled={page === 1} onClick={() => setPage(p => p - 1)} className="h-7 w-7 rounded flex items-center justify-center hover:bg-accent disabled:opacity-50">‹</button>
                <button className="h-7 w-7 rounded bg-primary text-primary-foreground flex items-center justify-center font-bold">{page}</button>
                <button onClick={() => setPage(p => p + 1)} className="h-7 w-7 rounded flex items-center justify-center hover:bg-accent">›</button>
              </div>
            </div>
          </>
          )}
        </div>

        {/* Detail Panel */}
        {detailLoading && !detailData && (
           <div className="bg-card rounded-xl p-6 flex items-center justify-center min-h-[400px]">
             <Loader2 size={24} className="animate-spin text-muted-foreground" />
           </div>
        )}
        
        {!detailLoading && detailData && (
          <div className="bg-card rounded-xl p-6 space-y-5">
            <div className="flex items-start justify-between">
              <div>
                {Math.abs(detailData.bias_mean) > 1.5 && <span className="bg-primary/10 text-primary text-[10px] font-bold tracking-wider px-2 py-0.5 rounded uppercase">High Bias Warning</span>}
                <h3 className="text-xl font-bold font-display mt-2">{detailData.merchant_name}</h3>
                <p className="text-xs text-muted-foreground">{detailData.merchant_id} · {detailData.city}</p>
              </div>
              <button onClick={() => setSelectedId(null)} className="text-muted-foreground hover:text-foreground">
                <X size={18} />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-accent rounded-lg p-3">
                <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">Orders/Day</span>
                <div className="text-lg font-extrabold font-display text-primary">{detailData.avg_daily_orders}</div>
              </div>
              <div className="bg-accent rounded-lg p-3">
                <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">Bias Score</span>
                <div className="text-lg font-extrabold font-display">{detailData.bias_mean.toFixed(2)}</div>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">KPT Distribution</span>
                <span className="text-xs text-primary font-medium">Last 100 Orders</span>
              </div>
              <div className="h-[120px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={kptDistData}>
                    <XAxis dataKey="range" tickLine={false} axisLine={false} tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
                    <Bar dataKey="count" fill="var(--color-primary)" radius={[3, 3, 0, 0]} opacity={0.7} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold tracking-widest text-muted-foreground uppercase">Bias Trend</span>
                <span className="flex items-center gap-1 text-xs"><span className="h-2 w-2 rounded-full bg-primary animate-pulse" /> Live</span>
              </div>
              <div className="h-[100px] bg-siren-light rounded-lg p-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={biasTrendData}>
                    <Line type="monotone" dataKey="bias" stroke="var(--color-primary)" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <button className="w-full bg-foreground text-background font-bold py-3 rounded-lg flex items-center justify-center gap-2 hover:opacity-90 transition-opacity">
              <Flag size={16} /> Flag for Audit
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
