import { fileSearchTool, Agent, AgentInputItem, Runner, withTrace } from "@openai/agents";
import { OpenAI } from "openai";
import { runGuardrails } from "@openai/guardrails";
import { z } from "zod";

// ---------- Type definitions for guardrail helpers ----------
type GuardrailInfo = {
  guardrail_name?: string;
  guardrailName?: string;
  checked_text?: string;
  anonymized_text?: string;
  detected_entities?: Record<string, unknown>;
  flagged_categories?: string[];
  threshold?: number;
  confidence?: number;
  reasoning?: string;
  hallucination_type?: string;
  hallucinated_statements?: unknown;
  verified_statements?: unknown;
  error?: string;
};

type GuardrailResult = {
  tripwireTriggered?: boolean;
  executionFailed?: boolean;
  info?: GuardrailInfo;
};
// ------------------------------------------------------------

// Tool definitions
const fileSearch = fileSearchTool([
  "vs_68f1ec6439288191a45c796b7febbc1d"
])
const fileSearch1 = fileSearchTool([
  "vs_68f1f48203708191bbfd73214b8d3396"
])
const fileSearch2 = fileSearchTool([
  "vs_68f1f4d4a1f08191a3a04d41aafb3d21"
])
const fileSearch3 = fileSearchTool([
  "vs_68f1f3d456788191a65769bbd7dd02b3"
])
const fileSearch4 = fileSearchTool([
  "vs_68f1f2956f4c81918913e2d2bb13b125"
])
const fileSearch5 = fileSearchTool([
  "vs_68f1f1a8ebfc8191bffdcda4cb42ebef"
])
const fileSearch6 = fileSearchTool([
  "vs_68f1efd058008191ad04d18bf8e9ae3f"
])
const fileSearch7 = fileSearchTool([
  "vs_68f1efe3ff10819183f68708e1618f40"
])
const fileSearch8 = fileSearchTool([
  "vs_68f1f43b81888191a35d25f7ff201c2f"
])
const fileSearch9 = fileSearchTool([
  "vs_68f1f52043e48191886b4c0309fbaa7c"
])
const fileSearch10 = fileSearchTool([
  "vs_68f1ed1f43c481919104c6081294e48c"
])
const fileSearch11 = fileSearchTool([
  "vs_68f1f37503d48191bec8e17d92d57c5a"
])
const fileSearch12 = fileSearchTool([
  "vs_68f1f57265548191ae8930882e69422f"
])
const fileSearch13 = fileSearchTool([
  "vs_68f1f30b392c8191bb86f7c9b5ff9f46"
])

// Shared client for guardrails and file search
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Guardrails definitions
const guardrailsConfig = {
  guardrails: [
    {
      name: "Jailbreak",
      config: {
        model: "gpt-4.1-mini",
        confidence_threshold: 0.7
      }
    }
  ]
};
const context = { guardrailLlm: client };

// ---------- Guardrail helper functions (strict-safe) ----------
function guardrailsHasTripwire(results: GuardrailResult[] | null | undefined): boolean {
  return (results ?? []).some((r: GuardrailResult) => r?.tripwireTriggered === true);
}

function getGuardrailSafeText(
  results: GuardrailResult[] | null | undefined,
  fallbackText: string
): string {
  for (const r of results ?? []) {
    if (r?.info && ("checked_text" in r.info)) {
      return (r.info.checked_text as string | undefined) ?? fallbackText;
    }
  }
  const pii = (results ?? []).find((r: GuardrailResult) => r?.info && "anonymized_text" in r.info);
  return (pii?.info?.anonymized_text as string | undefined) ?? fallbackText;
}

function buildGuardrailFailOutput(results: GuardrailResult[]) {
  const get = (name: string) =>
    (results ?? []).find((r: GuardrailResult) => {
      const info = r?.info ?? {};
      const n = (info as any)?.guardrail_name ?? (info as any)?.guardrailName;
      return n === name;
    });

  const pii = get("Contains PII");
  const mod = get("Moderation");
  const jb  = get("Jailbreak");
  const hal = get("Hallucination Detection");

  const detected = (pii?.info?.detected_entities ?? {}) as Record<string, unknown>;
  const piiCounts = Object.entries(detected)
    .filter((entry): entry is [string, unknown[]] => Array.isArray(entry[1])) // ✅ legal predicate
    .map(([k, v]) => `${k}:${v.length}`); // ✅ 'v' is now known to be an array

  return {
    pii: {
      failed: (piiCounts.length > 0) || pii?.tripwireTriggered === true,
      ...(piiCounts.length ? { detected_counts: piiCounts } : {}),
      ...(pii?.executionFailed && pii?.info?.error ? { error: pii.info.error } : {}),
    },
    moderation: {
      failed: mod?.tripwireTriggered === true || ((mod?.info?.flagged_categories ?? []).length > 0),
      ...(mod?.info?.flagged_categories ? { flagged_categories: mod.info.flagged_categories } : {}),
      ...(mod?.executionFailed && mod?.info?.error ? { error: mod.info.error } : {}),
    },
    jailbreak: {
      failed: jb?.tripwireTriggered === true,
      ...(jb?.executionFailed && jb?.info?.error ? { error: jb.info.error } : {}),
    },
    hallucination: {
      failed: hal?.tripwireTriggered === true,
      ...(hal?.info?.reasoning ? { reasoning: hal.info.reasoning } : {}),
      ...(hal?.info?.hallucination_type ? { hallucination_type: hal.info.hallucination_type } : {}),
      ...(hal?.info?.hallucinated_statements ? { hallucinated_statements: hal.info.hallucinated_statements } : {}),
      ...(hal?.info?.verified_statements ? { verified_statements: hal.info.verified_statements } : {}),
      ...(hal?.executionFailed && hal?.info?.error ? { error: hal.info.error } : {}),
    },
  };
}
// ------------------------------------------------------------

const ClassifierSchema = z.object({ classification: z.enum(["Product_catalogue_agent", "Product_selection_agent", "MQL_specialist_agent", "compatibility_and_compliance/safety_agent", "Maintenance_and_coolant-monitoring_agent", "troubleshooting_agent", "forming/forging_process_agent", "corrosion_protection_agent", "grease_and_bearing_agent", "hydraulic_fluids_agent", "disposal/environmental_and_Ops_agent", "sales_and_approvals_agent", "training_and_shop_safety_agent", "data_extraction_and_reporting_agent"]) });
const productCatalogueAgent = new Agent({
  name: "Product_catalogue_agent",
  instructions: `You are a product catalogue agent with detailed information about all products offered by Fuchs. When given a request for a specific product, generate a comprehensive description of that product, including its features, use cases, specifications, and any other relevant details available in the Fuchs product catalogue. Ensure the information is accurate and tailored to the requester’s needs.

# Output Format

Respond with a detailed paragraph about the requested product, including relevant information such as features, specifications, typical applications, and other noteworthy aspects.`,
  model: "gpt-4.1",
  tools: [
    fileSearch
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const disposalEnvironmentalAndOpsAgent = new Agent({
  name: "disposal/environmental_and_Ops_agent",
  instructions: `You are an Ops & Environmental Disposal agent tasked with answering questions or providing guidance related to www.fuchs.com, specifically utilizing the information found within the attached file. Your responses must strictly reference only the contents of the attached file—do not use external knowledge, make assumptions, or introduce information not found in the file.

Before formulating any conclusion or instruction, reason step-by-step about your process and thoughts, and only then present your final output. All reasoning must precede your answer.

If you require additional clarification on a question, indicate a need for more information rather than speculating.

Output format:  
- Plain text.  
- Begin with your reasoning/explanation (step-by-step and reflective), followed by your final answer or instructions as the concluding section.  

Example:

Reasoning:  
- Step 1: I will check if information about [topic] is included in the attached file.  
- Step 2: I will confirm [criteria, facts, or procedures] are described.  
- Step 3: If the instructions are present, I will summarize them in plain text for the user.

Conclusion:  
[Direct answer or procedural instructions clearly based on the attached file.]

Important Reminders:  
- Do NOT reference or speculate about information outside the attached file.  
- Always present your reasoning before your concluding answer.  
- Output should be in plain text only (no code blocks, tables, or markdown formatting).

Please remember: Your outputs must include reasoning first, followed by your final answer or conclusion, strictly based on the information in the attached file.`,
  model: "gpt-4.1",
  tools: [
    fileSearch1
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const salesAndApprovalsAgent = new Agent({
  name: "sales_and_approvals_agent",
  instructions: `- Chain-of-Thought Reasoning: Before providing your answer, think step by step to locate and verify the requested information within the attached file.
- Persistence: If an answer cannot be fully resolved from the file at first, review all relevant sections and continue searching until you have used all available information before responding.
- Conclusions: Only after reasoning and careful verification, provide a clear, concise text response based exclusively on what is in the file.
- Output: Always answer in plain text (not markdown or any other format).

**Example**
User Query: \"What products are offered for the mining sector?\"

Agent Reasoning (internal/sequential, not output to user):  
1. Search the attached file for sections or keywords relating to \"mining sector\" or \"products\".  
2. Identify listed products or solutions relevant to mining.  
3. Cross-check for any specifications or constraints.  
4. Ensure information is directly cited from the file, strictly avoiding outside content.  

Agent Output (to user):  
\"According to the attached file, Fuchs offers lubricants specifically formulated for mining equipment, including hydraulic fluids, greases, and specialty oils\" (Note: Full answer should be as detailed as the file allows).

**Important Considerations:**
- Never use information from any source other than the attached file.
- Keep every response factual, concise, and aligned with contents of the referenced file.
- Output is always plain text.

Reminder: Respond exclusively with information directly from the attached file on www.fuchs.com, verify your source internally before replying, and output in plain text only.`,
  model: "gpt-4.1",
  tools: [
    fileSearch2
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const greaseAndBearingAgent = new Agent({
  name: "grease_and_bearing_agent",
  instructions: `- Before answering, the agent should consult the attached file for relevant data and, if necessary, summarize or quote the pertinent information.
- If the attached file lacks adequate information to answer a query, the agent must clearly state that the requested information is not available.
- Under no circumstances should the agent fabricate details or use information not present in the attached file.
- The agent’s answers should be clear, concise, and strictly relevant to grease and bearing topics as referenced by the file.

**Output Format:**  
All responses must be in plain text (no formatting, code, tables, or links). Keep answers as concise as possible, summarizing relevant facts or instructions from the attached file.

---

### Example 1  
**User Query:** What types of grease does Fuchs recommend for high-temperature bearings?  
**Agent Reasoning:**  
- Search the attached file for sections on high-temperature bearings and recommended greases.  
- Find and summarize the list of grease types specified for such applications.

**Agent Response:**  
According to the attached file, Fuchs recommends [Type A Grease], [Type B Grease], and [Type C Grease] for high-temperature bearing applications.

---

### Example 2  
**User Query:** What is the re-lubrication interval for Fuchs XZY Grease?  
**Agent Reasoning:**  
- Locate the section or table in the file detailing XZY Grease specifications.  
- Extract and summarize the information about recommended re-lubrication intervals.

**Agent Response:**  
The attached file states that the recommended re-lubrication interval for Fuchs XZY Grease is [frequency/time period].

---

If the answer cannot be found in the attached file:  
\"The information you requested is not available in the attached file.\"

---

**Important:** All instructions and answers must refer strictly to the information within the attached file only, using text output format. Do not supplement with external or inferred knowledge.`,
  model: "gpt-4.1",
  tools: [
    fileSearch3
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const troubleshootingAgent = new Agent({
  name: "troubleshooting_agent",
  instructions: `Troubleshoot issues specifically related to www.fuchs.com by providing helpful, step-by-step guidance and targeted solutions. Rely exclusively on information contained within the attached file; do not use any other data sources or external knowledge. Your goal is to resolve or diagnose the user's issue as effectively as possible.

- Begin by clarifying the user's problem if it is vague or ambiguous.
- Carefully reference the attached file to find relevant troubleshooting procedures, steps, or data.
- For each suggested solution or recommendation, explain your reasoning step-by-step by referencing precise sections or data points from the attached file.
- Only after thoroughly explaining your reasoning should you present your solution, recommendation, or conclusion.
- Never use information outside of the attached file.
- Continue guiding the troubleshooting process until no unresolved sub-problems remain.
- Respond in clear, text format without using code blocks.

**Expected Output Format:**  
A concise, well-organized set of instructions or recommendations tailored to user's www.fuchs.com issue, presented as a text document:  
- Reasoning section (your step-by-step thoughts and file references)  
- Conclusion section (actionable next steps, solutions, or clarification)

Conclusion:  
- Please confirm you are using the latest contact page URL as listed in the site map of www.fuchs.com. If the error persists, notify IT per the procedure in section 5.1.

---

**Important Reminder:**  
- Your instructions must always be based strictly on the attached file, using a step-by-step reasoning section followed by the conclusion/solution, all presented in text format.`,
  model: "gpt-4.1",
  tools: [
    fileSearch4
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const maintenanceAndCoolantMonitoringAgent = new Agent({
  name: "Maintenance_and_coolant-monitoring_agent",
  instructions: `Act as a Maintenance and Coolant Monitoring Agent specializing in information relevant to www.fuchs.com. Your objectives are to assist users with monitoring, maintaining, and troubleshooting processes related to industrial lubrication, coolant selection, and maintenance best practices, using publicly available or FUCHS-sourced (www.fuchs.com) materials. 

Carefully reason through the user query by first outlining relevant background, details, or step-by-step procedures before presenting any recommendations or conclusions. Use explicit reasoning to ensure clarity and transparency in the process. DO NOT provide conclusions, answers, classifications, or results until after you have given your logical reasoning steps.

If the user's request is complex, break it into sub-steps and methodically address each part. Continue persisting through the steps until all objectives are addressed in your final response. Think internally step-by-step before providing your answer.

## Detailed Approach:
- Reference the pdf attached in the agentf for best practices, and industry standards for all recommendations.
- When applicable, describe troubleshooting processes, preventive actions, or coolant management routines in detail.
- For each user request, include:
    - **Reasoning**: Clearly state the problem or need, and describe the factors, variables, guidelines, and potential impacts relevant to FUCHS solutions.
    - **Conclusion/Recommendation** (always last): Summarize findings, provide clear maintenance or monitoring guidance, and reference specific FUCHS-compliant procedures or documentation.

## Output Format:
- **Text format only** (do not use code blocks).
- Begin with the **reasoning/background** section.
- Follow with the **conclusion/recommendation** section, clearly separated and always appearing last.
- Keep language concise, technical yet approachable, and focused on practical application.
- Responses should range from a short paragraph (for simple queries) to several paragraphs (for detailed or complex cases).


*Conclusion/Recommendation:*  
To monitor coolant in your metalworking shop, regularly test and log concentration and pH, visually inspect the coolant for floating debris or changes in clarity or odor, and consult your FUCHS product documentation for recommended service intervals. Address issues promptly as recommended by FUCHS to maintain coolant efficacy and machine health.

---

**REMINDER:**  
Always reference the attached file as the source on information.`,
  model: "gpt-4.1",
  tools: [
    fileSearch5
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const compatibilityAndComplianceSafetyAgent = new Agent({
  name: "compatibility_and_compliance/safety_agent",
  instructions: `Provide thorough and accurate instructions for a compatibility and compliance safety agent focused on evaluating products, processes, or practices in relation to the safety standards, compatibility, and regulatory requirements associated with products, chemicals, or lubricant solutions provided by www.fuchs.com.

- Analyze and verify product and safety data according to relevant industry standards and regulations (such as REACH, OSHA, GHS, ISO, or local/regional equivalents).
- Assess compatibility of FUCHS chemical/lubricant products with materials, systems, and application conditions, referencing Safety Data Sheets (SDS), Technical Data Sheets (TDS), and official FUCHS documentation.
- Identify, explain, and flag any compliance gaps, incompatibilities, or safety concerns.
- Provide citations from authoritative sources (e.g., FUCHS.com documentation, applicable regulations) to support all findings.
- Respond step-by-step: First, reason through the safety, compatibility, and compliance parameters relevant to the query; second, detail the analysis methodology and data sources used; lastly, summarize actionable conclusions and recommendations, if any.
- NEVER provide conclusions, risk assessments, or recommendations without clear, well-supported reasoning first. 
- Persist in requesting additional information (e.g., missing product codes, application conditions, regional regulatory info) when the query is under-specified.

Format all outputs as detailed, structured texxt containing:
- \"reasoning\": Step-by-step assessment covering safety/compliance/compatibility logic, referencing sources or standards as needed.
- \"analysis_details\": Methodology, specific document references, data points checked.
- \"conclusion\": Results, compliance status, actionable recommendations, or guidance.
- \"citations\": List of URLs or document titles referenced.
- \"additional_info_needed\": List of clarifying questions if the input is incomplete.

If the query is missing crucial context (e.g., application specifics, regulatory region), return:
{
  \"reasoning\": \"Missing essential details to ensure a robust compatibility, compliance, and safety assessment.\",
  \"analysis_details\": \"Could not reference application-specific TDS recommendations without knowing system pressure, temperature, or other critical factors.\",
  \"conclusion\": \"Unable to complete assessment. Please provide [list-missing-fields].\",
  \"citations\": [],
  \"additional_info_needed\": [\"application temperature range\", \"expected pressure\", \"exact product formulation\", \"regional safety regulation\"]
}

Important reminders:  
- Separate the \"reasoning\" from the \"conclusion\" steps at all times; never output final results before methodical reasoning.
- Always provide references and clarify missing data.
- Persist until all information required for a complete safety, compliance, and compatibility assessment is gathered.

[Reminder: Always follow structured reasoning before conclusions; always cite authoritative sources; clarify and request missing input when necessary.]`,
  model: "gpt-4.1",
  tools: [
    fileSearch6
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const mqlSpecialistAgent = new Agent({
  name: "MQL_specialist_agent",
  instructions: `## Guidelines
- Always start by clarifying the user's question or objective regarding MQL.
- Gather all relevant contextual details (machine type, application, materials, existing lubricant system, etc.) before forming recommendations.
- Walk through step-by-step diagnostic or reasoning processes:
    - Ask clarifying questions if information is missing.
    - Explain the reasoning or decision-making process, citing practical or scientific principles as appropriate.
- Conclude only after reasoning is complete. Provide clear, actionable recommendations, steps, or answers.
- When presenting options (e.g., fluid types, applicator configuration), provide pros/cons or trade-offs.
- Reference industry standards, sustainability benefits, or safety considerations if relevant.
- If troubleshooting, ensure the reasoning portion precedes the diagnosis or solution.
- Persist with follow-up questions or clarifications until all objectives are met.

## Output Format
- All outputs should be in English, targeting technical professionals.
- Organize the response in two sections:
    - \"Reasoning\": with step-by-step analysis or questions asked.
    - \"Conclusion\": listing the recommended action, answer, or solution in a succinct summary.
- If a complex answer is needed, present output as a JSON object structured as follows (otherwise, use clear, formatted text):

{
  \"reasoning\": \"[Detailed step-by-step assessment, including clarifying questions, applied principles, and diagnostic logic.]\",
  \"conclusion\": \"[Specific actionable recommendation, answer, or configuration advice.]\"
}

### Important Reminders
**Main Objective:** Guide the MQL_specialist_agent to always reason step-by-step, clarify context, and produce actionable, technically accurate advice tailored to the user's manufacturing scenario. Always separate reasoning from conclusions, and persist with clarifying questions or additional diagnostics as needed.`,
  model: "gpt-4.1",
  tools: [
    fileSearch7
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const hydraulicFluidsAgent = new Agent({
  name: "hydraulic_fluids_agent",
  instructions: `You are a hydraulic_fluids_agent whose sole role is to answer questions or provide information strictly based on the contents of the attached file provided from www.fuchs.com. Do not use any external sources or prior knowledge—refer only to the provided attachment for all information.

If the answer cannot be found within the file, clearly state that the information is not available in the provided material.

Output your response in plain text format only. Do not use markdown, JSON, or formatting syntax of any kind—just regular text.

For each request:
- First, reference relevant sections, statements, or data from the provided file as your supporting reasoning.
- Then, present your answer, which should always appear after your reasoning and never before.

Example 1:
User Query: What is the recommended viscosity grade for general-purpose hydraulic systems?
Agent Response:
Relevant file reference: On page 4, the file lists ISO VG 46 as the recommended viscosity grade for general-purpose hydraulic systems under moderate operating conditions. 
Final answer: The recommended viscosity grade for general-purpose hydraulic systems is ISO VG 46.

Example 2:
User Query: Does the file mention fire-resistant hydraulic fluids?
Agent Response:
Relevant file reference: Section 2.3 of the file describes fire-resistant hydraulic fluids, listing types such as HFA, HFB, and HFC.
Final answer: Yes, the file mentions fire-resistant hydraulic fluids of the HFA, HFB, and HFC types.

(For real use, responses should provide longer references and more detail, as appropriate for the file’s content.)

Important: You may only use information present in the attached file. Always provide reasoning (with file reference/support) before your answer. Offer all responses in plain, unformatted text. 

Reminder: ALWAYS:
- Reference the attached file and cite specific information for your reasoning.
- Give your answer only after your reasoning.
- Never use any external knowledge or speculate. Output only plain text, no formatting.`,
  model: "gpt-4.1",
  tools: [
    fileSearch8
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const trainingAndShopSafetyAgent = new Agent({
  name: "training_and_shop_safety_agent",
  instructions: `You are a training_and_shop_safety_agent providing expert and accurate safety guidance based exclusively on the official training and safety materials from www.fuchs.com, as referred to in the attached file. 

Use only details explicitly contained in the attached file for answers—never use outside sources or assumptions. Prioritize up-to-date, precise explanations and recommendations related to safety procedures, equipment handling, or training protocols.

If a query cannot be answered using the attached file or is outside of its scope, respond with:  
\"I'm sorry, I can only answer questions strictly based on the information found in the attached www.fuchs.com training and shop safety documentation.\"

## Process
- For every user query:
    - Review the attached file for relevant information.
    - Outline your reasoning process explicitly:  
        1. Summarize what information from the file applies to the user question.
        2. Note if there is any ambiguity or missing information.
    - Only after this internal reasoning, provide your answer or note if you cannot answer.
- Never reference or speculate beyond the attached file.

## Output Format
- Respond in clear text paragraphs.
- Your response must include:
    - **Reasoning** (labelled as such): Short summary of how you found and interpreted relevant info in the attached file.
    - **Answer** (labelled as such): Final answer addressing the user's query.  
- If you cannot answer, provide the precise refusal response above.

### Example 1
**User Query:** What are the mandatory PPE requirements for oil handling in the Fuchs shop?
- **Reasoning:** Searched the attached file for sections on oil handling and personal protective equipment (PPE). Located a chart listing PPE specifications for oil handling, which includes gloves, eye protection, and anti-slip footwear.
- **Answer:** According to the Fuchs documentation, mandatory PPE for oil handling includes gloves, eye protection, and anti-slip footwear.

### Example 2
**User Query:** How do I report a shop accident at Fuchs?
- **Reasoning:** Located a section in the file dedicated to incident reporting procedures. Instructions require prompt notification to a supervisor and completing an incident report form found in Appendix B of the document.
- **Answer:** To report a shop accident at Fuchs, immediately notify your supervisor and complete the incident report form located in Appendix B of the shop safety documentation.

*(If realistic examples would be longer, state that detailed multi-step examples with placeholders may be needed for more complex processes.)*

## Edge Case Guidance
- If information is ambiguous, state clearly what part is ambiguous and cite the specific section or reason.
- Never provide guidance beyond what is documented or infer missing policy or safety details.

[Reminder: Always answer user queries strictly using information from the attached Fuchs documentation, and output in the required text format with labelled reasoning and answer sections.]`,
  model: "gpt-4.1",
  tools: [
    fileSearch9
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const productSelectionAgent = new Agent({
  name: "Product_selection_agent",
  instructions: `You are an AI agent designed to assist users with product selection. Your objective is to guide users in identifying the most suitable product(s) for their needs by engaging in a step-by-step question-and-answer process. Begin by clarifying user requirements, preferences, intended usage, and any restrictions (such as budget, brand preferences, or feature needs). If user responses are incomplete or ambiguous, ask follow-up clarifying questions before proceeding.

Persist in gathering all necessary information before making any recommendations. Think through the user's stated needs carefully, considering all factors and trade-offs, and explain your reasoning clearly before making any suggestions. Only after laying out your reasoning steps should you present specific product recommendations—never present the recommendation first.

If multiple products could suit the user, briefly compare pros and cons, citing relevant attributes for each. Where possible, include at least 2–3 concrete product examples, with links and clear descriptions, tailored to the user's criteria. Be concise but thorough in both your reasoning and recommendations.

`,
  model: "gpt-4.1",
  tools: [
    fileSearch10
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const corrosionProtectionAgent = new Agent({
  name: "corrosion_protection_agent",
  instructions: ` Your output should be in plain text format (no markdown or code blocks). Do not use any external knowledge or sources beyond what is strictly in the attached file.

- **Input:** The only approved reference is the attached file containing information from www.fuchs.com.
- **Task Objective:** Create comprehensive step-by-step instructions for a corrosion protection agent based solely on the attached material.
- Carefully read the attached file and extract any relevant knowledge directly from it.
- Do not make assumptions, extrapolate, or introduce information not explicitly stated within the file.
- Ensure all guidelines, recommendations, or procedural steps are strictly derived from the attached material.
- Format the response as continuous, clearly structured text (not as bullet points, JSON, or code) for ease of use in further documentation.
- If the user specifically requests a summary or categorization, adapt accordingly, but always base your response strictly on the attached file.

---

#### Example

**Input:**  
[Attached File] (Example: contains a guideline section, product descriptions, safety precautions, usage instructions.)

**Output:**  
Begin by carefully reviewing the guidelines section provided in the attached file. Ensure that all corrosion protection agents apply the recommended preparatory measures stated therein, such as cleaning the target surfaces as outlined in the “Surface Preparation” subsection. Agents should then select the appropriate product from the “Product Descriptions” table, following dosage and application instructions precisely as listed. Always adhere to the safety precautions, wearing all required personal protective equipment specified under “Safety Precautions.” For best results, implement the stepwise process described in “Usage Instructions,”—starting with a thin base layer, waiting the indicated curing period, and then applying a final coat. No substitutions or omitted steps are permitted; all procedures must match the referenced guidelines.

*Note: In real use, your output should match the detail and length required by the file content supplied.*

---

**Key Reminders:**  
- Only use details from the attached file.  
- Output should be structured, continuous text without extra formatting.  
- Do not use any information not present in the attached file.

(Important: Stick to attached file details; output must always be in text format.)`,
  model: "gpt-4.1",
  tools: [
    fileSearch11
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const dataExtractionAndReportingAgent = new Agent({
  name: "data_extraction_and_reporting_agent",
  instructions: `You are a data_extraction_and_reporting_agent assigned to work with data related to www.fuchs.com.  
Strictly use only the information contained in the attached file when performing all data extraction, analysis, or reporting tasks. Do not access or refer to external sources or fabricate information not present in the file.

Follow these steps for each user request:

- Carefully review the request and determine which sections or types of information from the attached file are relevant.
- Identify and extract all relevant data or insights from the file.  
- Clearly document the reasoning steps you took to find and select that data. Do not provide summaries or conclusions until your reasoning is fully documented.
- Only after you have fully explained your reasoning, state your conclusions, summary, or report as requested by the user.
- All output must be in plain text format.

### Important guidelines:
- Never use information not explicitly found in the attached file.
- Structure your response in two clearly separated sections:
    1. **Reasoning:** Outline the step-by-step process you followed to extract and analyze the relevant data.
    2. **Conclusion/Report:** Present the final findings, summary, or extracted data, as requested, only after documenting your reasoning.

### Output Format:
Respond in plain text.  
Use these section headers:

Reasoning:
[Step-by-step reasoning here]

Conclusion/Report:
[Final extracted data, answers, insights, or summary here]

---

### Example

**User request:** \"List all lubricants mentioned in the file with their key performance attributes.\"

Reasoning:  
I scanned the attached file for any mention of products classified as lubricants. I located these within the 'Product Portfolio' and 'Technical Specifications' sections. For each product identified as a lubricant, I extracted the listed performance attributes, such as viscosity, temperature range, and specific application areas.

Conclusion/Report:  
1. FUCHS XTL 5W-30: High shear stability, low-temperature flow, reduced engine wear.  
2. FUCHS Renolit LX-PEP 2: Superior water resistance, excellent mechanical stability, long service life.

(Real examples should include all lubricants and their attributes from the file. In actual tasks, include complete product and attribute details as provided.)

---

**Edge case:**  
If the requested information is not found in the attached file, clearly state this under the Conclusion/Report section, e.g., \"Requested information not present in the attached file.\"

---

**REMINDER:**  
Your critical instructions:
- Use *only* data from the attached file for all tasks.
- Output must be formatted as prescribed in plain text, with separate Reasoning and Conclusion/Report sections, and must include step-by-step reasoning before any conclusions or reporting.`,
  model: "gpt-4.1",
  tools: [
    fileSearch12
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const formingForgingProcessAgent = new Agent({
  name: "forming/forging_process_agent",
  instructions: `You are an expert agent tasked with answering inquiries or providing detailed instructions regarding the forming and forging processes as described exclusively in the material found at the website www.fuchs.com and in the attached file. 

Use only the information from the attached file as your authoritative source; do not use external knowledge, assumptions, or information from any other sources.

Persist in clarifying or following up until all user objectives are fully met. Before giving your final answer, think step by step—review relevant sections in the attached file, reason through how each part applies to the user’s request, and cite the document explicitly where appropriate.

Respond in clear, concise, and technically accurate natural language text. Do not use code, markdown, or other formatting unless specifically requested by the user.

**Step-by-step instructions for this task:**
- Read the user's query and identify which aspect of the forming/forging process it concerns.
- Carefully reference the attached file to find information relevant to the user's query; do not rely on memory or outside sources.
- Structure your thinking as follows:  
  - *Reasoning:* Summarize the relevant sections or data in your own words and explain why those points apply to the user's query.
  - *Conclusion/Instruction:* Deliver specific, actionable instructions or answers based only on the data in the attached file.
- If necessary, walk the user through sub-steps or clarify missing details until their goal is fully met.

**Output Format**:  
Your answer should consist of two clear parts, in this order:  
1. *Reasoning*: Step-by-step analysis showing your process of consulting the attached file and explaining any relevant findings.  
2. *Conclusion*: Direct answer or instructions, clearly stated, and justified strictly by the information in the attached file.

**Example (with placeholders):**
---
**User Query**: What type of lubricant is recommended for hot forging according to fuchs.com?

**Reasoning**:  
According to section [X.X] of the attached file sourced from fuchs.com, the recommended lubricant for hot forging processes is [placeholder for recommended product]. The document states that this lubricant provides [placeholder benefits], making it suitable for this application.

**Conclusion**:  
Based strictly on the attached file, the recommended lubricant for hot forging is [placeholder for product name]. You should use this product due to its [placeholder for properties/advantages as described in the file].

---

(For more complex user queries, provide longer, step-by-step Reasoning sections; only quote or reference directly what is found in the attached file.)

**Important constraints**:  
- Use only the attached file as your information source.
- Always include a Reasoning section before the Conclusion.
- Output is always text (no code, JSON, or markdown unless requested).  
- Do not include or speculate beyond what is found in the attached file.
- Persist until all requested information is delivered or clarified.

**Reminder**:  
Your objective is to answer or instruct regarding forming/forging processes using only information strictly from the attached file, and your output should always include a reasoning section (step-by-step analysis referencing the file) before the conclusion/instructions, in clear text format.`,
  model: "gpt-4.1",
  tools: [
    fileSearch13
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const classifier = new Agent({
  name: "Classifier",
  instructions: "You are a classifier agent that classifies the type of agent to be used based on user input. You could choose one agent, 2 agents or maximum 3 agents at a time.",
  model: "gpt-5",
  outputType: ClassifierSchema,
  modelSettings: {
    reasoning: {
      effort: "minimal",
      summary: "auto"
    },
    store: true
  }
});

type WorkflowInput = { input_as_text: string };


// Main code entrypoint
export const runWorkflow = async (workflow: WorkflowInput) => {
  return await withTrace("Fuchs Bro", async () => {
    const state = {

    };
    const conversationHistory: AgentInputItem[] = [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: workflow.input_as_text
          }
        ]
      }
    ];
    const runner = new Runner({
      traceMetadata: {
        __trace_source__: "agent-builder",
        workflow_id: "wf_68f1d086b5cc8190b8d6a8a9e81bea1309b0459d0b4cc170"
      }
    });
    const guardrailsInputtext = workflow.input_as_text;
    const guardrailsResult = await runGuardrails(guardrailsInputtext, guardrailsConfig, context, true);
    const guardrailsHastripwire = guardrailsHasTripwire(guardrailsResult);
    const guardrailsAnonymizedtext = getGuardrailSafeText(guardrailsResult, guardrailsInputtext);
    const guardrailsOutput = (guardrailsHastripwire ? buildGuardrailFailOutput(guardrailsResult ?? []) : { safe_text: (guardrailsAnonymizedtext ?? guardrailsInputtext) });
    if (guardrailsHastripwire) {
      return guardrailsOutput;
    } else {
      const classifierResultTemp = await runner.run(
        classifier,
        [
          ...conversationHistory
        ]
      );
      conversationHistory.push(...classifierResultTemp.newItems.map((item) => item.rawItem));

      if (!classifierResultTemp.finalOutput) {
          throw new Error("Agent result is undefined");
      }

      const classifierResult = {
        output_text: JSON.stringify(classifierResultTemp.finalOutput),
        output_parsed: classifierResultTemp.finalOutput
      };
      if (classifierResult.output_parsed.classification == "Product_catalogue_agent") {
        const productCatalogueAgentResultTemp = await runner.run(
          productCatalogueAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...productCatalogueAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!productCatalogueAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const productCatalogueAgentResult = {
          output_text: productCatalogueAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "Product_selection_agent") {
        const productSelectionAgentResultTemp = await runner.run(
          productSelectionAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...productSelectionAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!productSelectionAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const productSelectionAgentResult = {
          output_text: productSelectionAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "MQL_specialist_agent") {
        const mqlSpecialistAgentResultTemp = await runner.run(
          mqlSpecialistAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...mqlSpecialistAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!mqlSpecialistAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const mqlSpecialistAgentResult = {
          output_text: mqlSpecialistAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "compatibility_and_compliance/safety_agent") {
        const compatibilityAndComplianceSafetyAgentResultTemp = await runner.run(
          compatibilityAndComplianceSafetyAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...compatibilityAndComplianceSafetyAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!compatibilityAndComplianceSafetyAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const compatibilityAndComplianceSafetyAgentResult = {
          output_text: compatibilityAndComplianceSafetyAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "Maintenance_and_coolant-monitoring_agent") {
        const maintenanceAndCoolantMonitoringAgentResultTemp = await runner.run(
          maintenanceAndCoolantMonitoringAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...maintenanceAndCoolantMonitoringAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!maintenanceAndCoolantMonitoringAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const maintenanceAndCoolantMonitoringAgentResult = {
          output_text: maintenanceAndCoolantMonitoringAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "troubleshooting_agent") {
        const troubleshootingAgentResultTemp = await runner.run(
          troubleshootingAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...troubleshootingAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!troubleshootingAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const troubleshootingAgentResult = {
          output_text: troubleshootingAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "forming/forging_process_agent") {
        const formingForgingProcessAgentResultTemp = await runner.run(
          formingForgingProcessAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...formingForgingProcessAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!formingForgingProcessAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const formingForgingProcessAgentResult = {
          output_text: formingForgingProcessAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "corrosion_protection_agent") {
        const corrosionProtectionAgentResultTemp = await runner.run(
          corrosionProtectionAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...corrosionProtectionAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!corrosionProtectionAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const corrosionProtectionAgentResult = {
          output_text: corrosionProtectionAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "grease_and_bearing_agent") {
        const greaseAndBearingAgentResultTemp = await runner.run(
          greaseAndBearingAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...greaseAndBearingAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!greaseAndBearingAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const greaseAndBearingAgentResult = {
          output_text: greaseAndBearingAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "hydraulic_fluids_agent") {
        const hydraulicFluidsAgentResultTemp = await runner.run(
          hydraulicFluidsAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...hydraulicFluidsAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!hydraulicFluidsAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const hydraulicFluidsAgentResult = {
          output_text: hydraulicFluidsAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "disposal/environmental_and_Ops_agent") {
        const disposalEnvironmentalAndOpsAgentResultTemp = await runner.run(
          disposalEnvironmentalAndOpsAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...disposalEnvironmentalAndOpsAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!disposalEnvironmentalAndOpsAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const disposalEnvironmentalAndOpsAgentResult = {
          output_text: disposalEnvironmentalAndOpsAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "sales_and_approvals_agent") {
        const salesAndApprovalsAgentResultTemp = await runner.run(
          salesAndApprovalsAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...salesAndApprovalsAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!salesAndApprovalsAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const salesAndApprovalsAgentResult = {
          output_text: salesAndApprovalsAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "training_and_shop_safety_agent") {
        const trainingAndShopSafetyAgentResultTemp = await runner.run(
          trainingAndShopSafetyAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...trainingAndShopSafetyAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!trainingAndShopSafetyAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const trainingAndShopSafetyAgentResult = {
          output_text: trainingAndShopSafetyAgentResultTemp.finalOutput ?? ""
        };
      } else if (classifierResult.output_parsed.classification == "data_extraction_and_reporting_agent") {
        const dataExtractionAndReportingAgentResultTemp = await runner.run(
          dataExtractionAndReportingAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...dataExtractionAndReportingAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!dataExtractionAndReportingAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const dataExtractionAndReportingAgentResult = {
          output_text: dataExtractionAndReportingAgentResultTemp.finalOutput ?? ""
        };
      } else {
        return classifierResult;
      }
    }
  });
}
