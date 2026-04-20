const API_BASE = "http://localhost:8000/api";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Failed to fetch health");
  return res.json();
}

export async function fetchDashboardSummary() {
  const res = await fetch(`${API_BASE}/dashboard/summary`);
  if (!res.ok) throw new Error("Failed to fetch dashboard summary");
  return res.json();
}

export async function fetchLiveOrders(limit = 50, city = "all", status = "all") {
  const params = new URLSearchParams({ limit: limit.toString(), city, status });
  const res = await fetch(`${API_BASE}/orders/live?${params.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch live orders");
  return res.json();
}

export async function fetchMerchants(params: { city?: string; cuisine?: string; tier?: string; behavior?: string; search?: string; page?: number; per_page?: number } = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined) query.append(key, value.toString());
  });
  const res = await fetch(`${API_BASE}/merchants?${query.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch merchants");
  return res.json();
}

export async function fetchMerchantDetail(merchantId: string) {
  const res = await fetch(`${API_BASE}/merchants/${merchantId}`);
  if (!res.ok) throw new Error("Failed to fetch merchant details");
  return res.json();
}

export async function fetchSignalQuality(dateFrom = "", dateTo = "") {
  const params = new URLSearchParams();
  if (dateFrom) params.append("date_from", dateFrom);
  if (dateTo) params.append("date_to", dateTo);
  const res = await fetch(`${API_BASE}/signal-quality?${params.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch signal quality");
  return res.json();
}

export async function fetchModelResults() {
  const res = await fetch(`${API_BASE}/model-results`);
  if (!res.ok) throw new Error("Failed to fetch model results");
  return res.json();
}

export async function fetchSimulation(contam = 40, theta = 90, dispatchOffset = 3.0) {
  const params = new URLSearchParams({
    contam: contam.toString(),
    theta: theta.toString(),
    dispatch_offset: dispatchOffset.toString(),
  });
  const res = await fetch(`${API_BASE}/simulation?${params.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch simulation results");
  return res.json();
}

export async function predictKpt(data: {
  merchant_id: string;
  cuisine: string;
  order_complexity: number;
  hour: number;
  day_of_week: number;
  google_busyness_index: number;
  rain_flag: number;
  rain_severity: number;
}) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to predict KPT");
  return res.json();
}
