export type SymbolMeta = {
  symbol: string;
  name: string;
  sectors: string[];
};

// Offline-Fallback (Backend nicht erreichbar) - dieselben 23 Symbole, die
// frueher COMPANY_SECTORS im Backend waren, jetzt nur noch als kleine
// Popular-Teilmenge des vollen NYSE+NASDAQ-Universums (siehe
// api/routes/analyze.py: search_symbols).
const LEGACY_ENTRIES: Record<string, { name: string; sectors: string[] }> = {
  AAPL: { name: "Apple Inc.", sectors: ["Tech", "Digital Media"] },
  MO: { name: "Altria Group Inc.", sectors: ["Tabak"] },
  GOOGL: { name: "Alphabet Inc.", sectors: ["Internet Services", "Advertising", "Technology"] },
  TSLA: { name: "Tesla Inc.", sectors: ["EV", "Energy", "Robotics", "AI"] },
  AMD: { name: "Advanced Micro Devices Inc.", sectors: ["Semiconductors"] },
  PYPL: { name: "PayPal Holdings Inc.", sectors: ["FinTech", "Digital Payments"] },
  NVDA: { name: "NVIDIA Corporation", sectors: ["Semiconductors", "Tech"] },
  NKE: { name: "Nike Inc.", sectors: ["Sporting Goods"] },
  UNH: { name: "UnitedHealth Group Inc.", sectors: ["Healthcare"] },
  XPEV: { name: "XPeng Inc.", sectors: ["EV", "AI", "Robotics"] },
  OCGN: { name: "Ocugen Inc.", sectors: ["Biotech", "Healthcare", "Pharma"] },
  UAA: { name: "Under Armour Inc.", sectors: ["Sporting Goods"] },
  BABA: {
    name: "Alibaba Group Holding Limited",
    sectors: ["E-Commerce", "Cloud Computing", "Digital Media", "FinTech", "Logistics"],
  },
  LUMN: { name: "Lumen Technologies Inc.", sectors: ["Telecommunications"] },
  TTWO: { name: "Take-Two Interactive Software Inc.", sectors: ["Gaming", "Game Publisher"] },
  BIDU: { name: "Baidu Inc.", sectors: ["Search Engine", "Cloud", "E-Commerce"] },
  JD: { name: "JD.com Inc.", sectors: ["AI", "Robotics", "E-Commerce"] },
  CRSP: { name: "CRISPR Therapeutics AG", sectors: ["Pharma", "Biotech"] },
  NVO: { name: "Novo Nordisk A/S", sectors: ["Pharma"] },
  NFLX: { name: "Netflix Inc.", sectors: ["Media", "Film", "Streaming"] },
  ILMN: { name: "Illumina Inc.", sectors: ["Biotech", "Genomics"] },
  BYD: { name: "BYD Company Limited", sectors: ["EV"] },
  SAP: { name: "SAP SE", sectors: ["Cloud Computing"] },
};

export const LOCAL_SYMBOLS: SymbolMeta[] = Object.entries(LEGACY_ENTRIES).map(([symbol, entry]) => ({
  symbol,
  name: entry.name,
  sectors: entry.sectors,
}));
