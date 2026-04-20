import { Link, useLocation } from "@tanstack/react-router";
import {
  LayoutDashboard,
  TrendingUp,
  Store,
  BarChart3,
  CheckSquare,
  Settings,
  HelpCircle,
} from "lucide-react";

const navItems = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/order-prediction", label: "Order Prediction", icon: TrendingUp },
  { to: "/merchant-management", label: "Merchant Management", icon: Store },
  { to: "/signal-quality", label: "Signal Quality", icon: BarChart3 },
  { to: "/model-results", label: "Model Results", icon: CheckSquare },
] as const;

const bottomItems = [
  { to: "/settings", label: "Settings", icon: Settings },
  { to: "/dashboard", label: "Support", icon: HelpCircle },
] as const;

export function AppSidebar() {
  const location = useLocation();

  return (
    <aside className="flex h-screen w-[220px] flex-col bg-card py-6 shrink-0">
      <div className="px-6 mb-8">
        <div className="flex items-center gap-2">
          <span className="text-2xl font-extrabold tracking-tight text-primary font-display">SwiftRoute</span>
        </div>
        <p className="text-[10px] font-semibold tracking-[0.15em] text-muted-foreground uppercase mt-0.5">
          Operational Intel
        </p>
      </div>

      <nav className="flex-1 flex flex-col gap-0.5 px-3">
        {navItems.map((item) => {
          const isActive = location.pathname === item.to;
          return (
            <Link
              key={item.to}
              to={item.to}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-siren-light text-primary"
                  : "text-sidebar-foreground hover:bg-accent"
              }`}
            >
              <item.icon size={18} strokeWidth={isActive ? 2.2 : 1.8} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border mx-3 my-2" />

      <div className="flex flex-col gap-0.5 px-3">
        {bottomItems.map((item) => {
          const isActive = location.pathname === item.to && item.label === "Settings";
          return (
            <Link
              key={item.label}
              to={item.to}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-siren-light text-primary"
                  : "text-sidebar-foreground hover:bg-accent"
              }`}
            >
              <item.icon size={18} strokeWidth={1.8} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </div>
    </aside>
  );
}
