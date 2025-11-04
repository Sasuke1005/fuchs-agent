// ====================================
// 1️⃣ Imports
// ====================================
import 'dotenv/config';
import express, { type Request, type Response } from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";
import { runWorkflow } from "./workflow.js";

// ====================================
// 2️⃣ App setup
// ====================================
const app = express();
app.use(express.json());
app.use(express.static("public")); // serves any static files if needed

// ✅ CORS allowlist – adjust these domains to match where your Fuchs UI is hosted
app.use(
  cors({ origin: ["https://chat.openai.com", "https://ai-pandit.com"] })
);

// ✅ Rate limiting (60 requests per minute per IP)
app.use(rateLimit({ windowMs: 60_000, max: 60 }));

// ✅ One-line request logging
app.use((req, _res, next) => {
  console.log(req.method, req.url, "ua:", req.headers["user-agent"]);
  next();
});

// ====================================
// 3️⃣ Routes
// ====================================

// Friendly GET for browser users
app.get("/ask", (_req, res) =>
  res
    .status(405)
    .json({ error: "Use POST /ask with JSON { message } and header X-Agent-Token" })
);

// Main POST endpoint
app.post("/ask", async (req: Request, res: Response) => {
  try {
    const token = String(req.headers["x-agent-token"] || "");
    const expected = process.env.AGENT_SERVER_TOKEN ?? "";
    if (!expected || token !== expected) {
      return res.status(401).json({ error: "unauthorized" });
    }

    const message = req.body?.message as string | undefined;
    if (!message) return res.status(400).json({ error: "message required" });

    // Run the fuchs-agents workflow
    const result = await runWorkflow({ input_as_text: message });

    // If the result contains output_text, normalize to "text"
    if (typeof result === "object" && "output_text" in result) {
      return res.json({
        text: (result as any).output_text,
        ...result,
        served_by: "fuchs-agent@render",
      });
    }

    // Otherwise, return whatever object the workflow produced
    return res.json({ ...result, served_by: "fuchs-agent@render" });
  } catch (e: any) {
    console.error(e);
    return res
      .status(500)
      .json({ error: "server_error", detail: String(e?.message || e) });
  }
});

// ====================================
// 4️⃣ Server start
// ====================================
const port = Number(process.env.PORT || 3000);
app.listen(port, () => console.log(`✅ Fuchs Agent running on :${port}`));