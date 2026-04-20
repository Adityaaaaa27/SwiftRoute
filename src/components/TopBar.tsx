import { Bell, User } from "lucide-react";

export function TopBar() {
  return (
    <header className="flex items-center justify-between px-6 py-3 bg-card">
      <div className="flex items-center gap-6">
        <span className="text-sm font-semibold text-primary font-display">SwiftRoute Operational Intel</span>
        <span className="text-sm font-medium text-primary underline underline-offset-4">City: Mumbai</span>
        <span className="text-sm text-muted-foreground">Date Range</span>
      </div>
      <div className="flex items-center gap-4">
        <button className="relative p-1.5 rounded-full hover:bg-accent transition-colors">
          <Bell size={18} className="text-foreground" />
          <span className="absolute top-0.5 right-0.5 h-2 w-2 rounded-full bg-primary" />
        </button>
        <div className="flex items-center gap-2">
          <div className="text-right text-xs">
            <div className="font-semibold text-foreground">Ops Lead</div>
            <div className="text-muted-foreground">Arjun Mehta</div>
          </div>
          <div className="h-8 w-8 rounded-full bg-accent flex items-center justify-center">
            <User size={16} className="text-muted-foreground" />
          </div>
        </div>
      </div>
    </header>
  );
}
