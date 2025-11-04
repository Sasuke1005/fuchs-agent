import {
  fileSearchTool,
  Agent,
  AgentInputItem,
  Runner,
  withTrace,
} from "@openai/agents";
import { OpenAI } from "openai";
import { runGuardrails } from "@openai/guardrails";
import { z } from "zod";

// ----------------------------------------------------------------------------
// Tool definitions
//
// The RustX agent uses a file search tool against a specific vector store.
// Replace the vector store ID below with your own if needed.
// ----------------------------------------------------------------------------
const fileSearch = fileSearchTool([
  "vs_690850e09c4c8191aec35b8135362fed",
]);

// Shared OpenAI client for guardrails and other API calls. The API key
// should be provided via an environment variable. Do not commit your key.
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Guardrails configuration. Here we enable a jailbreak detector using the
// gpt‑4.1‑mini model. Additional guardrails can be added as needed.
const guardrailsConfig = {
  guardrails: [
    {
      name: "Jailbreak",
      config: {
        model: "gpt-4.1-mini",
        confidence_threshold: 0.7,
      },
    },
  ],
};
const context = { guardrailLlm: client };

// ----------------------------------------------------------------------------
// Guardrails helper functions
// ----------------------------------------------------------------------------

function guardrailsHasTripwire(results: any[] | undefined): boolean {
  return (results ?? []).some((r) => r?.tripwireTriggered === true);
}

function getGuardrailSafeText(
  results: any[] | undefined,
  fallbackText: string
): string {
  // Prefer checked_text as the generic safe/processed text
  for (const r of results ?? []) {
    if (r?.info && "checked_text" in r.info) {
      return r.info.checked_text ?? fallbackText;
    }
  }
  // Fall back to PII-specific anonymized_text if present
  const pii = (results ?? []).find(
    (r) => r?.info && "anonymized_text" in r.info
  );
  return pii?.info?.anonymized_text ?? fallbackText;
}

function buildGuardrailFailOutput(results: any[] | undefined) {
  const get = (name: string) =>
    (results ?? []).find((r) => {
      const info = r?.info ?? {};
      const n = info?.guardrail_name ?? info?.guardrailName;
      return n === name;
    });
  const pii = get("Contains PII");
  const mod = get("Moderation");
  const jb = get("Jailbreak");
  const hal = get("Hallucination Detection");
  const piiCounts = Object.entries(pii?.info?.detected_entities ?? {})
    .filter(([, v]) => Array.isArray(v))
    .map(([k, v]) => `${k}:${(v as any[]).length}`);
  return {
    pii: {
      failed: piiCounts.length > 0 || pii?.tripwireTriggered === true,
      ...(piiCounts.length ? { detected_counts: piiCounts } : {}),
      ...(pii?.executionFailed && pii?.info?.error
        ? { error: pii.info.error }
        : {}),
    },
    moderation: {
      failed:
        mod?.tripwireTriggered === true ||
        ((mod?.info?.flagged_categories ?? []).length > 0),
      ...(mod?.info?.flagged_categories
        ? { flagged_categories: mod.info.flagged_categories }
        : {}),
      ...(mod?.executionFailed && mod?.info?.error
        ? { error: mod.info.error }
        : {}),
    },
    jailbreak: {
      failed: jb?.tripwireTriggered === true,
      ...(jb?.executionFailed && jb?.info?.error ? { error: jb.info.error } : {}),
    },
    hallucination: {
      failed: hal?.tripwireTriggered === true,
      ...(hal?.info?.reasoning ? { reasoning: hal.info.reasoning } : {}),
      ...(hal?.info?.hallucination_type
        ? { hallucination_type: hal.info.hallucination_type }
        : {}),
      ...(hal?.info?.hallucinated_statements
        ? { hallucinated_statements: hal.info.hallucinated_statements }
        : {}),
      ...(hal?.info?.verified_statements
        ? { verified_statements: hal.info.verified_statements }
        : {}),
      ...(hal?.executionFailed && hal?.info?.error
        ? { error: hal.info.error }
        : {}),
    },
  };
}

// ----------------------------------------------------------------------------
// Agent definitions
//
// A simple classifier distinguishes between catalogue vs product queries. You can
// extend the enum to support additional agent types if needed.
// ----------------------------------------------------------------------------
const ClassifierSchema = z.object({
  classification: z.enum(["Catalogue_Agent", "Product_Agent"]),
});

const classifier = new Agent({
  name: "Classifier",
  instructions:
    "You are a classifier agent whose role is to classify between different types of agents based on the user input.",
  model: "gpt-5",
  outputType: ClassifierSchema,
  modelSettings: {
    reasoning: {
      effort: "minimal",
      summary: "auto",
    },
    store: true,
  },
});

// Main RustX catalogue/product agent. It uses the fileSearch tool to consult
// your product catalogue and responds following strict guidelines on how to
// structure the answer. See the instructions string below for details.
const catalogueAndProductAgent = new Agent({
  name: "Catalogue and Product Agent",
  instructions: `Professionally respond to user queries about rustx (https://rustx.net/) using only information supported by the user’s context and the rustx catalogue. Always provide a complete, relevant, and concise answer to the main user query first. If the user asks about a specific product, structure your response using appropriate subheadings: "Why This Product," "Key Advantages," "Technical Aspects," "Safety and Compliance," "Value Proposition," and "Ordering and Action Steps." For general or non-specific queries, respond directly without using subheadings.

- Do not speculate or guess; use only verified catalogue or user-provided information.
- If you lack details to answer fully:
   - Clearly explain what additional information is needed.
   - Provide any partial, context-based answer possible.
   - After all possible information is given, end with a satisfaction or clarifying question.
- For complex/ambiguous queries, detail what information is missing before any clarifying or satisfaction question (always at the end).
- Always answer before asking for clarification or next steps.
- Persist until all objectives from the user query are addressed before concluding.
- Use chain-of-thought, step-by-step reasoning internally before finalizing your answer to ensure completeness and accuracy.

## Output Format
- For specific product queries: Use professional subheadings ("Why This Product," "Key Advantages," "Technical Aspects," "Safety and Compliance," "Value Proposition," "Ordering and Action Steps") as appropriate, followed by a single satisfaction/clarifying question at the end.
- For general queries: A single concise, direct paragraph fully answering the question, with an optional satisfaction/clarifying question at the end.
- Never begin with a question or request for clarification unless the catalogue provides absolutely no applicable information.

## Reasoning and Conclusion Order
- Reasoning: Gather and evaluate all available catalogue and user context, determining which subheadings and details apply based on the nature of the query.
- Conclusion: Present a full answer (with subheadings for specific products, direct response for general queries), then end with a satisfaction or follow-up question.

## Examples

**Example 1: General Category Query**  
User Input: "I need information on Rustx steel pipes."  
Agent Response:  
Rustx steel pipes are available in various diameters and specifications, suitable for multiple industrial applications. They feature high corrosion resistance and are manufactured according to international standards. Please let me know if you need details about a specific size, grade, or application, or if there is anything else I can help you with.

**Example 2: Product-Specific Query**  
User Input: "Tell me about Rustx SS 316 fasteners."  
Agent Response:  
**Why This Product**  
Rustx SS 316 fasteners are designed for use in environments requiring enhanced corrosion resistance, particularly in marine and chemical industries.  
**Key Advantages**  
They offer superior durability against pitting and crevice corrosion due to their molybdenum content.  
**Technical Aspects**  
Available in various sizes and thread patterns, these fasteners conform to international standards and endure harsh operational conditions.  
**Safety and Compliance**  
Manufactured according to ASTM standards, ensuring quality and safety compliance in demanding applications.  
**Value Proposition**  
Choosing Rustx SS 316 fasteners provides long-term reliability and reduces maintenance costs.  
**Ordering and Action Steps**  
Please specify the required dimensions or application details, and I can assist with pricing and availability. Would you like more information or a quotation tailored to your specifications?

**Example 3: Insufficient Information**  
User Input: "Can I get details on your wire mesh?"  
Agent Response:  
Rustx offers wire mesh in a variety of materials, sizes, and mesh configurations, compliant with relevant industry standards. To provide accurate details, could you specify the required material, mesh size, or intended application? Would you like to see available options or receive technical specifications for a particular use?

---

**Important Reminder:**  
Always address the user’s main query with catalogue evidence first. For specific product queries, use professional subheadings; for general queries, respond directly. Only ask for clarification or more information after fully answering based on available context. Never speculate or guess.
`,
  model: "gpt-4.1",
  tools: [fileSearch],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true,
  },
});

// ----------------------------------------------------------------------------
// Workflow definition
//
// The workflow runs guardrails on the input, classifies the user message, and
// delegates to the RustX catalogue agent for catalogue queries. If the
// classifier selects another category (e.g. Product_Agent), the classifier
// result is returned as-is. You can extend this logic by adding more agents.
// ----------------------------------------------------------------------------

export type WorkflowInput = { input_as_text: string };

export const runWorkflow = async (workflow: WorkflowInput) => {
  return await withTrace("RustX workflow", async () => {
    // Conversation history begins with the user message.
    const conversationHistory: AgentInputItem[] = [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: workflow.input_as_text,
          },
        ],
      },
    ];

    // Runner attaches trace metadata; update the workflow_id to match your
    // published workflow ID from Agent Builder. This is used for tracing in
    // Agent Builder dashboards.
    const runner = new Runner({
      traceMetadata: {
        __trace_source__: "agent-builder",
        workflow_id: "wf_69084e231b9481908d1b67cb2ad1963700715719a58eafbb",
      },
    });

    // ----------------------------------------------------------------------
    // Step 1: Guardrails evaluation
    // ----------------------------------------------------------------------
    const guardrailsInputtext = workflow.input_as_text;
    const guardrailsResult = await runGuardrails(
      guardrailsInputtext,
      guardrailsConfig,
      context,
      true
    );
    const hasTripwire = guardrailsHasTripwire(guardrailsResult);
    const anonymized = getGuardrailSafeText(
      guardrailsResult,
      guardrailsInputtext
    );
    const guardrailsOutput = hasTripwire
      ? buildGuardrailFailOutput(guardrailsResult)
      : { safe_text: anonymized ?? guardrailsInputtext };
    if (hasTripwire) {
      // If guardrails trip, return the guardrail result immediately
      return guardrailsOutput;
    }

    // ----------------------------------------------------------------------
    // Step 2: Classification
    // ----------------------------------------------------------------------
    const classifierResultTemp = await runner.run(classifier, [
      ...conversationHistory,
    ]);
    conversationHistory.push(
      ...classifierResultTemp.newItems.map((item) => item.rawItem)
    );
    if (!classifierResultTemp.finalOutput) {
      throw new Error("Classifier result is undefined");
    }
    const classifierResult = {
      output_text: JSON.stringify(classifierResultTemp.finalOutput),
      output_parsed: classifierResultTemp.finalOutput,
    };

    // ----------------------------------------------------------------------
    // Step 3: Route based on classification
    // ----------------------------------------------------------------------
    if (
      classifierResult.output_parsed.classification === "Catalogue_Agent"
    ) {
      // Run the RustX catalogue agent
      const catalogueResultTemp = await runner.run(catalogueAndProductAgent, [
        ...conversationHistory,
      ]);
      conversationHistory.push(
        ...catalogueResultTemp.newItems.map((item) => item.rawItem)
      );
      if (!catalogueResultTemp.finalOutput) {
        throw new Error("Catalogue agent result is undefined");
      }
      return {
        output_text: catalogueResultTemp.finalOutput,
        classification: classifierResult.output_parsed.classification,
      };
    } else {
      // If the classifier chose another category (e.g. Product_Agent), return
      // the classifier's output. You can extend this branch to call other
      // agents in the future.
      return classifierResult;
    }
  });
};