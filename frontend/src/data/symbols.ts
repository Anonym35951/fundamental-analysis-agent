export type SymbolMeta = {
  symbol: string;
  sectors: string[];
};

export const LOCAL_SYMBOLS: SymbolMeta[] = Object.entries({
  ILMN: ["Biotech", "Genomics"],
  GOOGL: ["Internet Services", "Advertising", "Technology"],
  TSLA: ["EV", "Energy", "Robotics", "AI"],
  AMD: ["Semiconductors"],
  PYPL: ["FinTech", "Digital Payments"],
  NVDA: ["Semiconductors", "Tech"],
  NKE: ["Sporting Goods"],
  UNH: ["Healthcare"],
  XPEV: ["EV", "AI", "Robotics"],
  OCGN: ["Biotech", "Healthcare", "Pharma"],
  UAA: ["Sporting Goods"],
  BABA: ["E-Commerce", "Cloud Computing", "Digital Media", "FinTech", "Logistics"],
  LUMN: ["Telecommunications"],
  TTWO: ["Gaming", "Game Publisher"],
  BIDU: ["Search Engine", "Cloud", "E-Commerce"],
  JD: ["AI", "Robotics", "E-Commerce"],
  CRSP: ["Pharma", "Biotech"],
  NVO: ["Pharma"],
  NFLX: ["Media", "Film", "Streaming"],
  AAPL: ["Tech", "Digital Media"],
  MO: ["Tabak"],
  BYD: ["EV"],
  SAP: ["Cloud Computing"]
}).map(([symbol, sectors]) => ({ symbol, sectors }));