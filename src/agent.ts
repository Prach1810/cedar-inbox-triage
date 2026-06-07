import Anthropic from "@anthropic-ai/sdk";
import "dotenv/config";
import {
  create_task,
  draft_message,
  escalate,
  find_slots,
  getToolCallsForItem,
  hold_slot,
  lookup_policy,
  search_patient,
  verify_insurance,
  withItemContext,
} from "./tools.js";
import type {
  Channel,
  Classification,
  Discipline,
  ExtractedIntake,
  InboxItem,
  ItemOutput,
  Patient,
  PolicyTopic,
  Slot,
  ToolResult,
  Urgency,
} from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Workflow =
  | "new_referral"
  | "safeguarding"
  | "scheduling"
  | "clinical_question"
  | "missing_paperwork"
  | "other";

interface TriagePlan {
  classification: Classification;
  urgency: Urgency;
  extracted_intake: ExtractedIntake;
  missing_info: string[];
  policy_topics: PolicyTopic[];
  workflow: Workflow;
  preferred_language: "en" | "es";
  recipient_hint: string | null;
  draft_channel: "email" | "portal" | "phone";
}

interface Narrative {
  recommended_next_action: string;
  draft_reply: string | null;
  decision_rationale: string;
}

interface OrchestrationFlags {
  recordDiscrepancy: boolean;
  oonBranch: boolean;
  slotsSkipped: boolean;
  patientMatch: Patient | null;
}

interface OrchestrationContext {
  toolResults: ToolResult<unknown>[];
  task_ids: string[];
  escalation: { reason: string; severity: "P0" | "P1" } | null;
  flags: OrchestrationFlags;
}

const SAFEGUARDING_PATTERNS =
  /\b(rough|abuse|abused|neglect|unsafe|violence|violent|hit|hitting|hurt|hurting)\b/i;

const MODEL = "claude-sonnet-4-6";

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  const results: ItemOutput[] = [];
  for (const item of inbox) {
    results.push(await withItemContext(item.id, () => triageItem(item)));
  }
  return results;
}

async function triageItem(item: InboxItem): Promise<ItemOutput> {
  const safeguarding = detectSafeguardingSignals(item);
  const plan = await extractTriagePlan(item);
  applySafetyOverrides(plan, safeguarding, item);

  const orchestration = await orchestrateWorkflow(item, plan);
  const toolSummary = summarizeToolResults(
    orchestration.toolResults,
    orchestration.flags,
  );
  const narrative = await synthesizeNarrative(item, plan, toolSummary);

  if (narrative.draft_reply) {
    await draft_message({
      recipient: resolveRecipient(plan, item),
      channel: plan.draft_channel,
      body: narrative.draft_reply,
      language: plan.preferred_language,
    });
  }

  return {
    item_id: item.id,
    classification: plan.classification,
    urgency: plan.urgency,
    requires_human_review: true,
    extracted_intake: plan.extracted_intake,
    missing_info: plan.missing_info,
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action: narrative.recommended_next_action,
    draft_reply: narrative.draft_reply,
    task_ids: orchestration.task_ids,
    escalation: orchestration.escalation,
    decision_rationale: narrative.decision_rationale,
  };
}

// ---------------------------------------------------------------------------
// Safety
// ---------------------------------------------------------------------------

function detectSafeguardingSignals(item: InboxItem): boolean {
  return SAFEGUARDING_PATTERNS.test(`${item.subject} ${item.body}`);
}

function applySafetyOverrides(
  plan: TriagePlan,
  safeguarding: boolean,
  item: InboxItem,
): void {
  sanitizeIntake(plan.extracted_intake);

  const detectedMissing = detectMissingInfo(
    plan.extracted_intake,
    `${item.subject} ${item.body}`,
  );
  if (detectedMissing.length > plan.missing_info.length) {
    plan.missing_info = detectedMissing;
  }

  if (safeguarding) {
    plan.classification = "safeguarding";
    plan.urgency = "P0";
    plan.workflow = "safeguarding";
    if (!plan.policy_topics.includes("safeguarding")) {
      plan.policy_topics.push("safeguarding");
    }
    return;
  }

  if (isMissingPaperworkCase(plan, item)) {
    plan.classification = "missing_paperwork";
    plan.urgency = "P2";
    plan.workflow = "missing_paperwork";
    return;
  }

  if (isSameDayScheduling(item)) {
    plan.classification = "scheduling";
    plan.urgency = "P1";
    plan.workflow = "scheduling";
  }
}

function isMissingPaperworkCase(plan: TriagePlan, item: InboxItem): boolean {
  const text = `${item.subject} ${item.body}`;
  if (/incomplete referral/i.test(item.subject)) return true;
  if ((text.match(/\[blank\]/gi) || []).length >= 2) return true;
  // Use core intake gaps only — not LLM-speculative missing_info (avoids false positives)
  const coreMissing = detectMissingInfo(plan.extracted_intake, text);
  return coreMissing.length >= 3;
}

function isSameDayScheduling(item: InboxItem): boolean {
  const text = `${item.subject} ${item.body}`.toLowerCase();
  const sameDay =
    text.includes("today") ||
    text.includes("can't make") ||
    text.includes("cannot make") ||
    /\b3\s*pm\b/.test(text);
  const schedulingAction =
    text.includes("reschedule") ||
    text.includes("re-schedule") ||
    text.includes("cancel") ||
    text.includes("cancellation");
  return sameDay && schedulingAction;
}

// ---------------------------------------------------------------------------
// LLM extraction
// ---------------------------------------------------------------------------

async function extractTriagePlan(item: InboxItem): Promise<TriagePlan> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    warnFallback(item.id, "no API key; using extraction template");
    return fallbackTriagePlan(item);
  }

  try {
    const client = new Anthropic({ apiKey });
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 1024,
      system: EXTRACTION_SYSTEM,
      messages: [
        {
          role: "user",
          content: `Extract triage plan as JSON only.\n\nChannel: ${item.channel}\nSender: ${item.sender}\nSubject: ${item.subject}\nBody: ${item.body}`,
        },
      ],
    });

    const text = response.content
      .filter((block) => block.type === "text")
      .map((block) => (block.type === "text" ? block.text : ""))
      .join("");

    const parsed = safeParseJson<Partial<TriagePlan>>(text);
    if (!parsed) {
      warnFallback(item.id, "extraction JSON parse failed; using template");
      return fallbackTriagePlan(item);
    }
    return normalizeTriagePlan(parsed, item);
  } catch (error) {
    warnFallback(
      item.id,
      `extraction API error: ${error instanceof Error ? error.message : "unknown"}`,
    );
    return fallbackTriagePlan(item);
  }
}

const EXTRACTION_SYSTEM = `You are a triage extraction assistant for Cedar Kids Therapy.
Return ONLY valid JSON (no markdown fences) matching this shape:
{
  "classification": "new_referral|existing_patient_request|scheduling|clinical_question|billing_question|missing_paperwork|provider_followup|complaint|safeguarding|spam|other",
  "urgency": "P0|P1|P2|P3",
  "extracted_intake": {
    "child_name": string|null,
    "dob_or_age": string|null,
    "parent_contact": string|null,
    "discipline": ["SLP"|"OT"|"PT"]|null,
    "diagnosis_or_concern": string|null,
    "payer": string|null,
    "member_id": string|null
  },
  "missing_info": string[],
  "policy_topics": ["service_lines"|"insurance"|"safeguarding"|"clinical_advice"|"scheduling"|"cancellation"|"language_access"],
  "workflow": "new_referral|safeguarding|scheduling|clinical_question|missing_paperwork|other",
  "preferred_language": "en"|"es",
  "recipient_hint": string|null,
  "draft_channel": "email"|"portal"|"phone"
}
Rules:
- Default urgency P2 unless safeguarding (P0) or same-day scheduling/cancellation (P1).
- Use null for unknown intake fields; list specific gaps in missing_info.
- Set preferred_language "es" for Spanish messages or Spanish preference.
- draft_channel: portal for portal_message, email for email, phone for fax/voicemail.
- Do not provide clinical advice.`;

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

async function orchestrateWorkflow(
  item: InboxItem,
  plan: TriagePlan,
): Promise<OrchestrationContext> {
  const ctx: OrchestrationContext = {
    toolResults: [],
    task_ids: [],
    escalation: null,
    flags: {
      recordDiscrepancy: false,
      oonBranch: false,
      slotsSkipped: false,
      patientMatch: null,
    },
  };

  switch (plan.workflow) {
    case "safeguarding":
      await runSafeguardingWorkflow(item, plan, ctx);
      break;
    case "scheduling":
      await runSchedulingWorkflow(plan, ctx);
      break;
    case "clinical_question":
      await runClinicalQuestionWorkflow(plan, ctx);
      break;
    case "missing_paperwork":
      await runMissingPaperworkWorkflow(plan, ctx);
      break;
    case "new_referral":
      await runNewReferralWorkflow(plan, ctx);
      break;
    default:
      await runOtherWorkflow(plan, ctx);
  }

  return ctx;
}

async function runSafeguardingWorkflow(
  item: InboxItem,
  plan: TriagePlan,
  ctx: OrchestrationContext,
): Promise<void> {
  ctx.toolResults.push(await lookup_policy({ topic: "safeguarding" }));

  const reason =
    "Disclosure suggesting possible harm or unsafe caregiving; mandated-reporter review required.";
  const esc = await escalate({
    item_id: item.id,
    reason,
    severity: "P0",
  });
  ctx.toolResults.push(esc);
  ctx.escalation = { reason, severity: "P0" };

  const task = await create_task({
    assignee: "clinical_lead",
    title: `Safeguarding review: ${plan.extracted_intake.child_name ?? "child"}`,
    due: todayDate(),
    notes:
      "Same-hour clinical lead review required. Do not send investigative questions to family.",
  });
  ctx.toolResults.push(task);
  ctx.task_ids.push(task.data.task_id);
}

async function runSchedulingWorkflow(
  plan: TriagePlan,
  ctx: OrchestrationContext,
): Promise<void> {
  const intake = plan.extracted_intake;
  const patientQuery = searchPatientArgs(intake);
  if (patientQuery.name || patientQuery.dob) {
    const search = await search_patient(patientQuery);
    ctx.toolResults.push(search);
    if (search.data.length > 0) {
      ctx.flags.patientMatch = search.data[0];
    }
  }

  ctx.toolResults.push(await lookup_policy({ topic: "scheduling" }));
  ctx.toolResults.push(await lookup_policy({ topic: "cancellation" }));

  const discipline = plan.extracted_intake.discipline?.[0] ?? "OT";
  const slots = await find_slots({
    discipline,
    preferences: "reschedule",
    language: plan.preferred_language,
  });
  ctx.toolResults.push(slots);

  const childRef = intake.child_name ?? "patient";
  const task = await create_task({
    assignee: "front_desk",
    title: `Reschedule request: ${childRef}`,
    due: todayDate(),
    notes: `Same-day reschedule request. Review available ${discipline} slots and contact family.`,
  });
  ctx.toolResults.push(task);
  ctx.task_ids.push(task.data.task_id);
}

async function runClinicalQuestionWorkflow(
  plan: TriagePlan,
  ctx: OrchestrationContext,
): Promise<void> {
  ctx.toolResults.push(await lookup_policy({ topic: "clinical_advice" }));

  const task = await create_task({
    assignee: "intake",
    title: `Clinical question follow-up: ${plan.extracted_intake.child_name ?? "family"}`,
    due: addDays(2),
    notes:
      "Parent asked a developmental question. Offer screening or evaluation; do not provide clinical advice in outbound message.",
  });
  ctx.toolResults.push(task);
  ctx.task_ids.push(task.data.task_id);
}

async function runMissingPaperworkWorkflow(
  plan: TriagePlan,
  ctx: OrchestrationContext,
): Promise<void> {
  const missing =
    plan.missing_info.length > 0
      ? plan.missing_info.join("; ")
      : "referral details incomplete";

  const task = await create_task({
    assignee: "intake",
    title: `Complete referral paperwork: ${plan.extracted_intake.child_name ?? "unknown child"}`,
    due: addDays(2),
    notes: `Missing: ${missing}. Contact referring provider or family.`,
  });
  ctx.toolResults.push(task);
  ctx.task_ids.push(task.data.task_id);
}

async function runNewReferralWorkflow(
  plan: TriagePlan,
  ctx: OrchestrationContext,
): Promise<void> {
  const intake = plan.extracted_intake;

  const patientQuery = searchPatientArgs(intake);
  if (patientQuery.name || patientQuery.dob) {
    const search = await search_patient(patientQuery);
    ctx.toolResults.push(search);
    if (search.data.length > 0) {
      ctx.flags.patientMatch = search.data[0];
      ctx.flags.recordDiscrepancy = guardianMismatch(
        search.data[0],
        intake.parent_contact,
      );
    }
  }

  const topics = uniqueTopics([
    "service_lines",
    "insurance",
    ...plan.policy_topics,
    ...(plan.preferred_language === "es" ? (["language_access"] as PolicyTopic[]) : []),
  ]);
  for (const topic of topics) {
    ctx.toolResults.push(await lookup_policy({ topic }));
  }

  let insuranceStatus: string | null = null;
  const payer = intake.payer;
  const memberId = intake.member_id;
  const canVerify =
    (payer && !isBlankPlaceholder(payer)) ||
    (memberId && !isBlankPlaceholder(memberId));
  if (canVerify) {
    const ins = await verify_insurance({
      payer: payer ?? undefined,
      member_id: memberId ?? undefined,
    });
    ctx.toolResults.push(ins);
    insuranceStatus = ins.data.status;

    if (insuranceStatus === "out_of_network" || insuranceStatus === "expired") {
      ctx.flags.oonBranch = true;
      ctx.flags.slotsSkipped = true;

      const childRef = intake.child_name ?? "patient";
      const task = await create_task({
        assignee: "billing",
        title: `Insurance review: ${childRef}`,
        due: addDays(2),
        notes: `${intake.payer ?? "Payer"} verified ${insuranceStatus}. Benefits conversation required before any slot hold.`,
      });
      ctx.toolResults.push(task);
      ctx.task_ids.push(task.data.task_id);
      return;
    }
  }

  const discipline = intake.discipline?.[0];
  if (discipline) {
    const slots = await find_slots({
      discipline,
      preferences: intake.diagnosis_or_concern ?? undefined,
      language: plan.preferred_language,
    });
    ctx.toolResults.push(slots);

    if (
      insuranceStatus === "in_network" &&
      slots.data.length > 0 &&
      intake.child_name
    ) {
      const first = slots.data[0] as Slot;
      const hold = await hold_slot({
        slot_id: first.slot_id,
        patient_ref: intake.child_name,
      });
      ctx.toolResults.push(hold);
    }
  }
}

async function runOtherWorkflow(
  plan: TriagePlan,
  ctx: OrchestrationContext,
): Promise<void> {
  ctx.toolResults.push(await lookup_policy({ topic: "service_lines" }));
  const task = await create_task({
    assignee: "intake",
    title: `Review inbox item: ${plan.extracted_intake.child_name ?? "unknown"}`,
    due: addDays(2),
    notes: "Unclassified item requires staff review.",
  });
  ctx.toolResults.push(task);
  ctx.task_ids.push(task.data.task_id);
}

// ---------------------------------------------------------------------------
// LLM synthesis
// ---------------------------------------------------------------------------

async function synthesizeNarrative(
  item: InboxItem,
  plan: TriagePlan,
  toolSummary: string,
): Promise<Narrative> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    warnFallback(item.id, "no API key; using narrative template");
    return fallbackNarrative(plan, toolSummary);
  }

  try {
    const client = new Anthropic({ apiKey });
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 1024,
      system: SYNTHESIS_SYSTEM,
      messages: [
        {
          role: "user",
          content: `Inbox item:\nSubject: ${item.subject}\nBody: ${item.body}\n\nTriage plan:\n${JSON.stringify(plan, null, 2)}\n\nTool results:\n${toolSummary}\n\nReturn JSON only with recommended_next_action, draft_reply, decision_rationale.`,
        },
      ],
    });

    const text = response.content
      .filter((block) => block.type === "text")
      .map((block) => (block.type === "text" ? block.text : ""))
      .join("");

    const parsed = safeParseJson<Partial<Narrative>>(text);
    const fallback = fallbackNarrative(plan, toolSummary);
    if (!parsed) {
      warnFallback(item.id, "synthesis JSON parse failed; using template");
      return fallback;
    }
    return {
      recommended_next_action:
        parsed.recommended_next_action ?? fallback.recommended_next_action,
      draft_reply: parsed.draft_reply ?? fallback.draft_reply,
      decision_rationale:
        parsed.decision_rationale ?? fallback.decision_rationale,
    };
  } catch (error) {
    warnFallback(
      item.id,
      `synthesis API error: ${error instanceof Error ? error.message : "unknown"}`,
    );
    return fallbackNarrative(plan, toolSummary);
  }
}

const SYNTHESIS_SYSTEM = `You write staff-facing triage narratives for Cedar Kids Therapy.
Return ONLY valid JSON: { "recommended_next_action": string, "draft_reply": string|null, "decision_rationale": string }
Rules:
- Ground all claims in the tool results provided.
- recommended_next_action must be a clear sentence for staff (not snake_case codes).
- draft_reply is a DRAFT only: never imply it was sent; never offer to schedule or confirm appointments.
- Offer that staff will follow up with options for human review instead of scheduling.
- No clinical advice in drafts; offer evaluation or staff follow-up instead.
- Safeguarding: neutral, empathetic acknowledgement only; no investigative questions.
- If preferred_language is "es", write draft_reply in Spanish.
- If recordDiscrepancy is noted, explain guardian mismatch in decision_rationale.`;

// ---------------------------------------------------------------------------
// summarizeToolResults
// ---------------------------------------------------------------------------

function summarizeToolResults(
  toolResults: ToolResult<unknown>[],
  flags: OrchestrationFlags,
): string {
  const lines: string[] = [];

  for (const tool of toolResults) {
    switch (tool.name) {
      case "search_patient": {
        const patients = tool.data as Patient[];
        if (patients.length === 0) {
          lines.push("- patient: no matches in system");
        } else {
          const p = patients[0];
          const disc = flags.recordDiscrepancy ? " (recordDiscrepancy: true)" : "";
          lines.push(
            `- patient: ${p.name}, guardian: ${p.guardian_name}, status: ${p.status}${disc}`,
          );
        }
        break;
      }
      case "verify_insurance": {
        const d = tool.data as { status: string; plan?: string };
        lines.push(
          `- insurance: ${d.status}${d.plan ? ` (${d.plan})` : ""}${tool.args.member_id ? `, member_id: ${tool.args.member_id}` : ""}`,
        );
        break;
      }
      case "lookup_policy": {
        const d = tool.data as { snippets: string[] };
        const topic = tool.args.topic as string;
        const preview = d.snippets[0]?.slice(0, 120) ?? "";
        lines.push(`- policy[${topic}]: ${preview}`);
        break;
      }
      case "find_slots": {
        const slots = tool.data as Slot[];
        if (slots.length === 0) {
          lines.push("- slots: 0 matching slots");
        } else {
          lines.push(
            `- slots: ${slots.length} matching; earliest ${slots[0].start} with ${slots[0].provider_name} (${slots[0].discipline})`,
          );
        }
        break;
      }
      case "hold_slot": {
        const d = tool.data as { hold_id: string; status: string; expires_at: string };
        lines.push(`- hold: ${d.hold_id} ${d.status} until ${d.expires_at}`);
        break;
      }
      case "create_task": {
        const d = tool.data as { task_id: string };
        lines.push(
          `- task: ${d.task_id} for ${tool.args.assignee} — ${tool.args.title}`,
        );
        break;
      }
      case "escalate": {
        lines.push(
          `- escalation: severity ${tool.args.severity} — ${tool.args.reason}`,
        );
        break;
      }
      default:
        lines.push(`- ${tool.name}: ${tool.result_summary}`);
    }
  }

  if (flags.oonBranch) {
    lines.push("- branch: OON/expired insurance — slot hold skipped per policy");
  }
  if (flags.slotsSkipped && !flags.oonBranch) {
    lines.push("- branch: slots not requested for this workflow");
  }
  if (flags.recordDiscrepancy && flags.patientMatch) {
    lines.push(
      `- recordDiscrepancy: message contact differs from system guardian (${flags.patientMatch.guardian_name})`,
    );
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Fallbacks
// ---------------------------------------------------------------------------

function fallbackTriagePlan(item: InboxItem): TriagePlan {
  const text = `${item.subject} ${item.body}`;
  const safeguarding = detectSafeguardingSignals(item);
  const sameDay = isSameDayScheduling(item);

  let classification: Classification = "new_referral";
  let urgency: Urgency = "P2";
  let workflow: Workflow = "new_referral";

  if (safeguarding) {
    classification = "safeguarding";
    urgency = "P0";
    workflow = "safeguarding";
  } else if (sameDay) {
    classification = "scheduling";
    urgency = "P1";
    workflow = "scheduling";
  } else if (/\[blank\]/i.test(text) || /incomplete referral/i.test(item.subject)) {
    classification = "missing_paperwork";
    workflow = "missing_paperwork";
  } else if (
    /is it normal|should i be worried|advice before booking/i.test(text)
  ) {
    classification = "clinical_question";
    workflow = "clinical_question";
  } else if (/referral/i.test(text)) {
    classification = "new_referral";
    workflow = "new_referral";
  }

  const intake = regexExtractIntake(text);
  const missing_info = detectMissingInfo(intake, text);
  const preferred_language = /espa[nñ]ol|hola|gracias|mensaje de voz/i.test(
    text,
  )
    ? "es"
    : "en";

  const policy_topics: PolicyTopic[] = ["service_lines"];
  if (intake.payer) policy_topics.push("insurance");
  if (safeguarding) policy_topics.push("safeguarding");
  if (workflow === "scheduling") policy_topics.push("scheduling", "cancellation");
  if (workflow === "clinical_question") policy_topics.push("clinical_advice");
  if (preferred_language === "es") policy_topics.push("language_access");

  return {
    classification,
    urgency,
    extracted_intake: intake,
    missing_info,
    policy_topics,
    workflow,
    preferred_language,
    recipient_hint: extractEmail(text) ?? extractPhone(text),
    draft_channel: defaultDraftChannel(item.channel),
  };
}

function fallbackNarrative(plan: TriagePlan, toolSummary: string): Narrative {
  const child = plan.extracted_intake.child_name ?? "your child";
  const contact = plan.extracted_intake.parent_contact ?? "the family";

  switch (plan.workflow) {
    case "safeguarding":
      return {
        recommended_next_action:
          "Clinical lead same-hour safeguarding review before any intake steps.",
        draft_reply:
          plan.preferred_language === "es"
            ? "Gracias por comunicarse con Cedar Kids Therapy. Hemos recibido su mensaje y un miembro de nuestro equipo se pondrá en contacto con usted pronto."
            : "Thank you for reaching out to Cedar Kids Therapy. We have received your message and a member of our team will follow up with you shortly.",
        decision_rationale:
          "Safeguarding disclosure detected; escalated to clinical lead per policy.",
      };
    case "scheduling":
      return {
        recommended_next_action:
          "Front desk to contact family today with reschedule options for staff review.",
        draft_reply: `Hi, thank you for letting us know about the change to today's appointment for ${child}. Our front desk team will follow up shortly with available options for your review.`,
        decision_rationale:
          "Same-day reschedule is a P1 operational issue. Patient lookup and slot search completed for staff review.",
      };
    case "clinical_question":
      return {
        recommended_next_action:
          "Intake to offer developmental screening or evaluation; no clinical advice via message.",
        draft_reply: `Thank you for your question about ${child}. We are not able to provide clinical advice over message, but we would be happy to discuss whether a screening or evaluation would be helpful. A team member will follow up soon.`,
        decision_rationale:
          "Clinical question routed per policy — acknowledge and offer evaluation pathway without giving clinical advice.",
      };
    case "missing_paperwork":
      return {
        recommended_next_action:
          "Intake to obtain missing referral fields before scheduling workflow.",
        draft_reply: `Thank you for the referral for ${child}. We need a few additional details (${plan.missing_info.join(", ") || "missing information"}) before we can proceed. Our intake team will follow up.`,
        decision_rationale:
          "Referral is incomplete; intake task created to gather missing paperwork.",
      };
    case "new_referral": {
      const oon =
        toolSummary.includes("out_of_network") ||
        toolSummary.includes("expired");
      const discrepancy = toolSummary.includes("recordDiscrepancy: true");
      return {
        recommended_next_action: oon
          ? "Billing to review insurance options with family before any slot hold."
          : "Intake to review referral, confirm insurance, and follow up on available evaluation slots.",
        draft_reply:
          plan.preferred_language === "es"
            ? `Hola, gracias por contactar a Cedar Kids Therapy sobre ${child}. Hemos recibido la solicitud de evaluación y nuestro equipo se comunicará con usted pronto para los siguientes pasos.`
            : oon
              ? `Hi, thank you for sending ${child}'s referral. Our billing team needs to review your insurance plan before we move forward with scheduling. A team member will follow up with options soon.`
              : `Hi, thank you for ${child}'s referral. We have received the request and our team will follow up shortly regarding next steps for an evaluation.`,
        decision_rationale: [
          oon
            ? "Insurance verification requires benefits conversation before scheduling."
            : "Standard new referral intake.",
          discrepancy
            ? "System-of-record guardian differs from message sender; staff should confirm contact."
            : "",
          toolSummary.includes("hold:")
            ? "Pending slot hold created for human review."
            : "",
        ]
          .filter(Boolean)
          .join(" "),
      };
    }
    default:
      return {
        recommended_next_action:
          "Intake to review and assign appropriate follow-up.",
        draft_reply: `Thank you for contacting Cedar Kids Therapy. We have received your message and a team member will follow up with ${contact} soon.`,
        decision_rationale:
          "Item requires staff review; standard acknowledgement drafted.",
      };
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseJsonResponse<T>(raw: string): T {
  let text = raw.trim();
  const fenceMatch = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  if (fenceMatch) {
    text = fenceMatch[1].trim();
  }
  const jsonStart = text.indexOf("{");
  const jsonEnd = text.lastIndexOf("}");
  if (jsonStart !== -1 && jsonEnd > jsonStart) {
    text = text.slice(jsonStart, jsonEnd + 1);
  }
  return JSON.parse(text) as T;
}

function safeParseJson<T>(raw: string): T | null {
  try {
    return parseJsonResponse<T>(raw);
  } catch {
    return null;
  }
}

function normalizeTriagePlan(
  partial: Partial<TriagePlan>,
  item: InboxItem,
): TriagePlan {
  const fallback = fallbackTriagePlan(item);
  return {
    classification: partial.classification ?? fallback.classification,
    urgency: partial.urgency ?? fallback.urgency,
    extracted_intake: {
      child_name:
        partial.extracted_intake?.child_name ??
        fallback.extracted_intake.child_name,
      dob_or_age:
        partial.extracted_intake?.dob_or_age ??
        fallback.extracted_intake.dob_or_age,
      parent_contact:
        partial.extracted_intake?.parent_contact ??
        fallback.extracted_intake.parent_contact,
      discipline:
        partial.extracted_intake?.discipline ??
        fallback.extracted_intake.discipline,
      diagnosis_or_concern:
        partial.extracted_intake?.diagnosis_or_concern ??
        fallback.extracted_intake.diagnosis_or_concern,
      payer:
        partial.extracted_intake?.payer ?? fallback.extracted_intake.payer,
      member_id:
        partial.extracted_intake?.member_id ??
        fallback.extracted_intake.member_id,
    },
    missing_info: partial.missing_info ?? fallback.missing_info,
    policy_topics: partial.policy_topics?.length
      ? partial.policy_topics
      : fallback.policy_topics,
    workflow: partial.workflow ?? fallback.workflow,
    preferred_language: partial.preferred_language ?? fallback.preferred_language,
    recipient_hint: partial.recipient_hint ?? fallback.recipient_hint,
    draft_channel: partial.draft_channel ?? fallback.draft_channel,
  };
}

function regexExtractIntake(text: string): ExtractedIntake {
  const childMatch =
    text.match(/Child:\s*([^.,\n]+)/i) ??
    text.match(/(?:son|daughter|hija|hijo)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i) ??
    text.match(/referral for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i);
  const dobMatch =
    text.match(/DOB[:\s]+(\d{4}-\d{2}-\d{2})/i) ??
    text.match(/DOB\s+(\d{4}-\d{2}-\d{2})/i);
  const parentMatch =
    text.match(/Parent:\s*([^\n.]+)/i) ??
    text.match(/I am (?:his|her) parent,\s*([^.\n]+)/i);
  const payerMatch = text.match(/Insurance[:\s]+([^.\n]+)/i);
  const memberMatch = text.match(/Member ID[:\s]+([A-Z0-9-]+)/i);
  const concernMatch = text.match(/Concern:\s*([^\n.]+)/i);

  let discipline: Discipline[] | null = null;
  if (/\bSLP\b|speech/i.test(text)) discipline = ["SLP"];
  else if (/\bOT\b|occupational/i.test(text)) discipline = ["OT"];
  else if (/\bPT\b|physical therapy|toe walking/i.test(text)) discipline = ["PT"];

  return {
    child_name: childMatch?.[1]?.trim() ?? null,
    dob_or_age: dobMatch?.[1]?.trim() ?? null,
    parent_contact: parentMatch?.[1]?.trim() ?? null,
    discipline,
    diagnosis_or_concern: concernMatch?.[1]?.trim() ?? null,
    payer: payerMatch?.[1]?.trim() ?? null,
    member_id: memberMatch?.[1]?.trim() ?? null,
  };
}

function detectMissingInfo(intake: ExtractedIntake, text: string): string[] {
  const missing: string[] = [];
  if (!intake.child_name || /Child:\s*\[blank\]/i.test(text))
    missing.push("child name");
  if (
    !intake.dob_or_age ||
    isBlankPlaceholder(intake.dob_or_age) ||
    /DOB:\s*\[blank\]/i.test(text)
  )
    missing.push("date of birth");
  if (
    !intake.parent_contact ||
    /Parent\/guardian:\s*\[blank\]/i.test(text)
  )
    missing.push("parent/guardian contact");
  if (
    !intake.payer ||
    isBlankPlaceholder(intake.payer) ||
    /Insurance:\s*\[blank\]/i.test(text)
  )
    missing.push("insurance payer");
  if (
    !intake.member_id ||
    isBlankPlaceholder(intake.member_id) ||
    /Member ID:\s*\[blank\]/i.test(text)
  )
    missing.push("insurance member ID");
  return missing;
}

function warnFallback(itemId: string, reason: string): void {
  console.warn(`[triage:${itemId}] ${reason}`);
}

function isBlankPlaceholder(value: string | null | undefined): boolean {
  if (!value) return true;
  const v = value.trim().toLowerCase();
  return v === "[blank]" || v === "blank" || v === "n/a" || v === "unknown";
}

function isValidDob(dob: string | null | undefined): boolean {
  return !!dob && /^\d{4}-\d{2}-\d{2}$/.test(dob.trim());
}

function searchPatientArgs(intake: ExtractedIntake): {
  name?: string;
  dob?: string;
} {
  const args: { name?: string; dob?: string } = {};
  if (intake.child_name) args.name = intake.child_name;
  if (isValidDob(intake.dob_or_age)) args.dob = intake.dob_or_age!.trim();
  return args;
}

function sanitizeIntake(intake: ExtractedIntake): void {
  if (isBlankPlaceholder(intake.payer)) intake.payer = null;
  if (isBlankPlaceholder(intake.member_id)) intake.member_id = null;
}

function guardianMismatch(
  patient: Patient,
  parentContact: string | null,
): boolean {
  if (!parentContact) return false;
  const guardian = patient.guardian_name.toLowerCase();
  const contact = parentContact.toLowerCase();
  const guardianParts = guardian.split(/\s+/);
  return !guardianParts.some(
    (part) => part.length > 2 && contact.includes(part),
  );
}

function resolveRecipient(plan: TriagePlan, item: InboxItem): string {
  const contact = plan.extracted_intake.parent_contact ?? "";
  const email =
    extractEmail(contact) ??
    extractEmail(item.body) ??
    extractEmail(item.sender);
  if (email) return email;

  const phone = extractPhone(contact) ?? extractPhone(item.body);
  if (phone) return phone;

  const hint = plan.recipient_hint;
  if (hint && (extractEmail(hint) || extractPhone(hint))) return hint;

  return item.sender;
}

function extractEmail(text: string): string | null {
  const match = text.match(/[\w.+-]+@[\w.-]+\.\w+/);
  return match?.[0] ?? null;
}

function extractPhone(text: string): string | null {
  const match = text.match(/\b555-\d{4}\b/);
  return match?.[0] ?? null;
}

function defaultDraftChannel(channel: Channel): "email" | "portal" | "phone" {
  if (channel === "portal_message") return "portal";
  if (channel === "email") return "email";
  return "phone";
}

function uniqueTopics(topics: PolicyTopic[]): PolicyTopic[] {
  return [...new Set(topics)];
}

function todayDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function addDays(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}
