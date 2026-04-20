// Mock data for the SIREN Dashboard

export const dashboardSummary = {
  averageKpt: { value: 14.2, unit: "min", change: "+2.4%", label: "vs LW" },
  riderWaitTime: { value: 4.8, unit: "min", change: "-0.8%", label: "vs LW" },
  p50EtaError: { value: 1.2, unit: "min", status: "Within Limit" },
  p90EtaError: { value: 5.4, unit: "min", status: "High Variance" },
  totalOrders: { value: "18.4k", change: "+14%", label: "Spike" },
  onTimeRate: { value: 94.2, unit: "%", status: "Optimal" },
};

export const kptTrendData = [
  { time: "08:00", actual: 12, predicted: 14 },
  { time: "10:00", actual: 15, predicted: 13 },
  { time: "12:00", actual: 18, predicted: 16 },
  { time: "14:00", actual: 14, predicted: 15 },
  { time: "16:00", actual: 16, predicted: 14 },
  { time: "18:00", actual: 13, predicted: 15 },
];

export const liveOrders = [
  { id: "#ZM-49021", merchant: "Leo's Pizzeria", cuisine: "Italian", predKpt: "12 MIN", status: "PREPARING", rider: "Rahul S.", riderDist: "(2.1km away)", signal: "clean" },
  { id: "#ZM-49028", merchant: "Green Kitchen", cuisine: "Continental", predKpt: "08 MIN", status: "ASSIGNED", rider: "Vikram K.", riderDist: "(On-site)", signal: "clean" },
  { id: "#ZM-49033", merchant: "Umami Bowl", cuisine: "Japanese", predKpt: "18 MIN", status: "PREPARING", rider: "Sameer L.", riderDist: "(Searching...)", signal: "suspicious" },
];

export const merchants = [
  { id: "#MZ-9921", name: "The Spice Hub", cuisine: "North Indian", tier: 1, city: "Mumbai", avgKpt: "14.2m", behavior: "Accurate", biasOffset: 0.12 },
  { id: "#MZ-4412", name: "Mainland China Express", cuisine: "Chinese", tier: 2, city: "Bangalore", avgKpt: "22.8m", behavior: "Late Marker", biasOffset: 0.84 },
  { id: "#MZ-7783", name: "Pizzeria Uno", cuisine: "Italian", tier: 1, city: "Delhi", avgKpt: "11.5m", behavior: "Early Marker", biasOffset: -0.32 },
  { id: "#MZ-2109", name: "South Taste Hub", cuisine: "South Indian", tier: 3, city: "Chennai", avgKpt: "18.4m", behavior: "Accurate", biasOffset: 0.05 },
  { id: "#MZ-3310", name: "Burger Empire", cuisine: "Fast Food", tier: 1, city: "Mumbai", avgKpt: "10.1m", behavior: "Early Marker", biasOffset: -0.18 },
  { id: "#MZ-8821", name: "Dosa Express", cuisine: "South Indian", tier: 2, city: "Pune", avgKpt: "16.7m", behavior: "Accurate", biasOffset: 0.22 },
];

export const signalQualityData = {
  layer1: {
    meanErrorReduction: -34.2,
    confidenceScore: 98.4,
    kptErrorBefore: [
      { range: "0-500", raw: 30, denoised: 10 },
      { range: "500-1000", raw: 45, denoised: 20 },
      { range: "1000-1500", raw: 35, denoised: 15 },
      { range: "1500-2000", raw: 20, denoised: 8 },
    ],
  },
  layer2: {
    rushMultiplier: 1.42,
    rainEvents: 12,
    thetaCurve: [
      { x: 0, y: 0.2 },
      { x: 20, y: 0.3 },
      { x: 40, y: 0.45 },
      { x: 60, y: 0.55 },
      { x: 80, y: 0.72 },
      { x: 100, y: 0.88 },
    ],
  },
  layer3: {
    classified: 82,
    behaviors: [
      { name: "Systematic Delay", value: 54, color: "#006D6F" },
      { name: "Early Handover", value: 28, color: "#B7122A" },
      { name: "Random Walk", value: 18, color: "#F4A5A5" },
    ],
    biasDistribution: [
      { range: "-3", value: 8 },
      { range: "-2", value: 15 },
      { range: "-1", value: 25 },
      { range: "0", value: 45 },
      { range: "1", value: 30 },
      { range: "2", value: 18 },
      { range: "3", value: 10 },
    ],
    skewness: 0.82,
    kurtosis: 3.12,
  },
};

export const simulationDefaults = {
  contamination: 14.2,
  theta: 0.82,
  riderOffset: 1.2,
  baselineMAE: 4.2,
  labelsFiltered: 1842,
  waitReduction: 2.4,
};

export const modelResults = {
  comparison: [
    { metric: "MAE (Mean Absolute Error)", baseline: "4.12 mins", siren: "2.84 mins", delta: "-31.06%" },
    { metric: "RMSE (Root Mean Square Error)", baseline: "6.58 mins", siren: "4.21 mins", delta: "-36.02%" },
    { metric: "R-Squared", baseline: "0.74", siren: "0.89", delta: "+20.27%" },
    { metric: "MAPE", baseline: "12.4%", siren: "8.1%", delta: "-34.67%" },
  ],
  featureImportance: [
    { name: "Historical Delivery Time", value: 94.2, category: "KINETIC" },
    { name: "Rider Availability Index", value: 88.5, category: "KINETIC" },
    { name: "Merchant Congestion Score", value: 76.1, category: "STATIC" },
    { name: "Precipitation Forecast", value: 52.4, category: "ENVIRONMENTAL" },
    { name: "Order Batching Factor", value: 41.8, category: "KINETIC" },
  ],
  tiers: [
    { id: "P1", name: "Ultra-Fast", target: "< 15 mins", mae: "1.2m", accuracy: "98.2%" },
    { id: "P2", name: "Standard", target: "15-30 mins", mae: "2.4m", accuracy: "94.1%" },
    { id: "P3", name: "Bulk/Sch", target: "> 30 mins", mae: "4.1m", accuracy: "86.5%" },
  ],
  scatterData: Array.from({ length: 50 }, (_, i) => ({
    actual: 5 + Math.random() * 30,
    predicted: 5 + Math.random() * 30,
  })),
  residualData: [
    { range: "-6", baseline: 5, siren: 3 },
    { range: "-4", baseline: 12, siren: 8 },
    { range: "-2", baseline: 25, siren: 15 },
    { range: "0", baseline: 35, siren: 45 },
    { range: "2", baseline: 20, siren: 12 },
    { range: "4", baseline: 10, siren: 5 },
    { range: "6", baseline: 4, siren: 2 },
  ],
  bottomStats: [
    { label: "TRAINING TIME", value: "14.2m", sub: "Optimized GPU Kernel" },
    { label: "SAMPLE SIZE", value: "1.8M", sub: "Last 30 Days" },
    { label: "MODEL DRIFT", value: "0.02%", sub: "Stable Distribution", highlight: true },
    { label: "INFERENCE LATENCY", value: "85ms", sub: "Real-time Hook" },
  ],
};
