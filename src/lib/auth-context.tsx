import { createContext, useContext, useState, type ReactNode } from "react";

interface User {
  email: string;
  role: "admin" | "ops_manager";
  name: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string, role: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    if (typeof window === "undefined") return null;
    const stored = localStorage.getItem("siren_user");
    return stored ? JSON.parse(stored) : null;
  });

  const login = async (email: string, _password: string, role: string) => {
    const u: User = {
      email,
      role: role as "admin" | "ops_manager",
      name: role === "admin" ? "Admin Console" : "Ops Lead",
    };
    setUser(u);
    localStorage.setItem("siren_user", JSON.stringify(u));
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem("siren_user");
  };

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
