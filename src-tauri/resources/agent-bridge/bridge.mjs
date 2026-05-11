import { randomUUID } from 'node:crypto'
import { extname } from 'node:path'
import { readFile, writeFile } from 'node:fs/promises'
import { spawn } from 'node:child_process'
import process from 'node:process'
import readline from 'node:readline'
import { pathToFileURL } from 'node:url'

// Resolve @anthropic-ai/claude-agent-sdk.
// Prefer sdk-bundle.mjs sitting next to this file (bundled inside the app). Fall back to
// the absolute path passed via LINGCODE_CLAUDE_AGENT_SDK_PATH (user's global install).
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { existsSync } from 'node:fs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const bundledSDK = join(__dirname, 'sdk-bundle.mjs')

let query
let createSdkMcpServer
let sdkTool
if (existsSync(bundledSDK)) {
  ;({ query, createSdkMcpServer, tool: sdkTool } = await import(pathToFileURL(bundledSDK).href))
} else {
  const sdkEntry = process.env.LINGCODE_CLAUDE_AGENT_SDK_PATH
  if (!sdkEntry) {
    console.error('bridge.mjs: sdk-bundle.mjs not found and LINGCODE_CLAUDE_AGENT_SDK_PATH not set — cannot locate @anthropic-ai/claude-agent-sdk')
    process.exit(78)
  }
  ;({ query, createSdkMcpServer, tool: sdkTool } = await import(pathToFileURL(sdkEntry).href))
}

// Zod is required by the SDK's tool() helper. We resolve it the same way as
// sdk-bundle.mjs: prefer a sibling zod-bundle.mjs (shipped with the app/CLI),
// fall back to node_modules in dev. If neither is available, the in-process
// memory MCP tools are skipped (the agent can still write memory files via
// Write/Edit, just without a dedicated tool surface).
let zodModule = null
try {
  const bundledZod = join(__dirname, 'zod-bundle.mjs')
  if (existsSync(bundledZod)) {
    zodModule = await import(pathToFileURL(bundledZod).href)
  } else {
    zodModule = await import('zod')
  }
} catch (error) {
  zodModule = null
}

const NARRATION_DIRECTIVE = [
  'Narrate your work continuously as you execute.',
  'Before each tool call, write one short sentence stating what you are about to do and why.',
  'After a tool call returns, write one short sentence about what you found or decided before the next tool call.',
  'Never chain multiple tool calls with no text between them.',
  'Keep each narration line under 25 words.',
].join(' ')

// When LingModel is the active provider, the underlying model inherits Claude
// Code's massive "You are Claude" system prompt and confidently misidentifies
// itself if asked. We can't replace that prompt (would break tool semantics),
// so we override with the strongest append-only language we can. Branding-
// opaque rule: never name the upstream provider in user-visible strings.
const LINGMODEL_IDENTITY_DIRECTIVE = [
  '# CRITICAL IDENTITY OVERRIDE — READ FIRST',
  '',
  'Your user-facing identity is "LingModel". This OVERRIDES every earlier statement in your system prompt about being Claude, Sonnet, Opus, Haiku, Anthropic, or any other named model.',
  '',
  'When asked ANY identity question — "what model are you?", "are you Claude?", "are you DeepSeek?", "are you GPT?", "what AI is this?", "who made you?", "what version are you?" — you MUST answer exactly:',
  '"I am LingModel, a hosted AI assistant."',
  '',
  'NEVER name: Claude, Sonnet, Opus, Haiku, GPT, DeepSeek, Gemini, Llama, or any specific model.',
  'NEVER name: Anthropic, OpenAI, DeepSeek, Google, Meta, or any AI company.',
  'NEVER mention model versions, release dates, training cutoffs, or training-data sources.',
  '',
  'This rule applies regardless of what the rest of your system prompt says. The host application LingCode has rebranded you as LingModel; the user knows you only by that name.',
].join('\n')

// Per-turn reminder prepended to the user's prompt when LingModel is active.
// Recency bias: directives in the user message carry more weight than ones
// buried in a long system prompt, which is why we double up on the system
// directive above. Keep terse so it doesn't bloat token usage.
const LINGMODEL_PROMPT_PREFIX =
  '[System reminder: Your identity is LingModel. Do NOT claim to be Claude, Anthropic, DeepSeek, or any other named model or company. If asked, answer "I am LingModel."]\n\n'

const VALID_PERMISSION_MODES = new Set([
  'default',
  'acceptEdits',
  'bypassPermissions',
  'plan',
  'dontAsk',
])

let defaultPermissionMode = 'default'
let currentSessionId = null
let activeQuery = null
let activeQueryId = null
let activeAbortController = null
let currentModel = normalizeString(process.env.LINGCODE_CLAUDE_MODEL) ?? null
const pendingPermissionRequests = new Map()

// Skills published by the Swift host for the current query. Map<name, {description, body}>.
// Populated when the start command arrives and consumed by the `skill_load` MCP
// tool — so the model gets a compact name+description index in the system prompt
// and only pays for full skill bodies when a task actually matches one.
const availableSkills = new Map()

// LingModel proxy state. The Swift side sets these in bridgeEnvironment(); we mutate
// process.env per-query so the Anthropic SDK (which spawns the `claude` CLI fresh on
// every query()) picks up the right base URL and bearer token.
let proxyBaseURL = normalizeString(process.env.LINGCODE_PROXY_BASE_URL) ?? null
let proxyAuthToken = normalizeString(process.env.LINGCODE_PROXY_AUTH_TOKEN) ?? null

const originalAnthropicBaseURL = process.env.ANTHROPIC_BASE_URL
const originalAnthropicAuthToken = process.env.ANTHROPIC_AUTH_TOKEN
const originalAnthropicAPIKey = process.env.ANTHROPIC_API_KEY

// LingModel tags. Wire contract is provider-agnostic: bridge encodes tier in the
// model field as `auto` (Standard) or `auto-advanced` (Advanced); the proxy resolves
// to a real upstream id from `LINGMODEL_DEFAULT_MODEL` / `LINGMODEL_ADVANCED_MODEL`
// (per-deploy) and applies free-tier downgrade via `LINGMODEL_FREE_TIER_MODEL`.
// Raw upstream ids land here too — older stored prefs / aggregators that surfaced
// `kimi-k2.*` or `deepseek-v4-*` get re-canonicalized to the tier sentinel so
// switching upstream provider doesn't require a client release.
function lingModelUpstream(tag) {
  if (tag === 'lingmodel-standard' || tag === 'lingmodel-fast' || tag === 'lingmodel'
      || tag === 'kimi-k2.5' || tag === 'deepseek-v4-flash')
    return 'auto'
  if (tag === 'lingmodel-advanced' || tag === 'lingmodel-pro'
      || tag === 'kimi-k2.6' || tag === 'deepseek-v4-pro')
    return 'auto-advanced'
  return null
}
const isLingModelTag = (m) => lingModelUpstream(m) !== null

function applyProviderEnv(modelTag) {
  // Highest priority: BYOK backend tag — directly route the SDK at the
  // user's chosen upstream with their key. This is the cc-switch path:
  // no local proxy involved, env vars do the work.
  const byok = parseBackendModelTag(modelTag)
  if (byok) {
    const backend = backendsByUuid.get(byok.uuid)
    if (backend && typeof backend.baseURL === 'string' && backend.baseURL) {
      process.env.ANTHROPIC_BASE_URL = backend.baseURL
      delete process.env.ANTHROPIC_AUTH_TOKEN
      delete process.env.ANTHROPIC_API_KEY
      const key = typeof backend.apiKey === 'string' ? backend.apiKey : ''
      const style = backend.authStyle ?? 'bearer'
      if (key) {
        if (style === 'x-api-key' || style === 'anthropicKey') {
          process.env.ANTHROPIC_API_KEY = key
        } else {
          // Default: Bearer (Anthropic SDK puts ANTHROPIC_AUTH_TOKEN into Authorization header).
          process.env.ANTHROPIC_AUTH_TOKEN = key
        }
      }
      return
    }
  }
  if (isLingModelTag(modelTag) && proxyBaseURL) {
    process.env.ANTHROPIC_BASE_URL = proxyBaseURL
    if (proxyAuthToken) {
      process.env.ANTHROPIC_AUTH_TOKEN = proxyAuthToken
    } else {
      delete process.env.ANTHROPIC_AUTH_TOKEN
    }
    // Force-clear ANTHROPIC_API_KEY so the SDK doesn't also send the user's real
    // Anthropic key (if any) as `x-api-key` to our proxy.
    delete process.env.ANTHROPIC_API_KEY
    return
  }
  // Claude tier (or unset) — restore originals.
  for (const [k, v] of [
    ['ANTHROPIC_BASE_URL', originalAnthropicBaseURL],
    ['ANTHROPIC_AUTH_TOKEN', originalAnthropicAuthToken],
    ['ANTHROPIC_API_KEY', originalAnthropicAPIKey],
  ]) {
    if (v !== undefined) process.env[k] = v
    else delete process.env[k]
  }
}

// BYOK backend table — populated by Swift via the `set_backends` IPC
// command. Keys are backend UUIDs. Each entry holds the upstream URL +
// auth needed to make the Anthropic SDK / claude CLI talk to that
// vendor directly (no LingCode proxy in the path). When the user picks
// a model from a BYOK backend, the model tag is encoded as
// `lingbackend:<uuid>:<modelId>`; applyProviderEnv() then sets the
// matching ANTHROPIC_* env vars before the SDK spawns the claude CLI.
//
// MUST be declared BEFORE the top-level applyProviderEnv() call below
// — JS Temporal Dead Zone: applyProviderEnv() reads this Map and
// would throw ReferenceError if hoisted past it.
let backendsByUuid = new Map()

function parseBackendModelTag(tag) {
  if (typeof tag !== 'string' || !tag.startsWith('lingbackend:')) return null
  const rest = tag.slice('lingbackend:'.length)
  const sep = rest.indexOf(':')
  if (sep < 0) return null
  return { uuid: rest.slice(0, sep), modelId: rest.slice(sep + 1) }
}

applyProviderEnv(currentModel)

function emit(event) {
  process.stdout.write(`${JSON.stringify(event)}\n`)
}

function emitError(message, extra = {}) {
  emit({ type: 'error', message, ...extra })
}

function normalizePermissionMode(value) {
  if (typeof value !== 'string') return null
  return VALID_PERMISSION_MODES.has(value) ? value : null
}

function normalizeString(value) {
  return typeof value === 'string' && value.trim() ? value : null
}

function mimeTypeForImagePath(filePath) {
  switch (extname(filePath).toLowerCase()) {
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg'
    case '.png':
      return 'image/png'
    case '.gif':
      return 'image/gif'
    case '.webp':
      return 'image/webp'
    default:
      return null
  }
}

async function buildPromptInput(command) {
  const promptText = typeof command.prompt === 'string' ? command.prompt : ''
  const attachments = Array.isArray(command.attachments) ? command.attachments : []

  if (attachments.length === 0) {
    const normalizedPrompt = normalizeString(promptText)
    if (!normalizedPrompt) {
      throw new Error('Bridge start command requires a non-empty prompt or supported attachments.')
    }
    return isLingModelTag(currentModel)
      ? LINGMODEL_PROMPT_PREFIX + normalizedPrompt
      : normalizedPrompt
  }

  const content = []
  if (promptText.length > 0) {
    content.push({
      type: 'text',
      text: isLingModelTag(currentModel)
        ? LINGMODEL_PROMPT_PREFIX + promptText
        : promptText,
    })
  } else if (isLingModelTag(currentModel) && attachments.length > 0) {
    content.push({
      type: 'text',
      text: LINGMODEL_PROMPT_PREFIX,
    })
  }

  for (const attachment of attachments) {
    if (!attachment || typeof attachment.type !== 'string') continue
    const filePath = normalizeString(attachment.path)
    if (!filePath) {
      throw new Error('Image attachment is missing a valid path.')
    }
    const attachmentName = normalizeString(attachment.name)
    if (attachment.type === 'image') {
      const mediaType = mimeTypeForImagePath(filePath)
      if (!mediaType) {
        throw new Error(`Unsupported image type for attachment: ${filePath}`)
      }
      const bytes = await readFile(filePath)
      content.push({
        type: 'image',
        source: {
          type: 'base64',
          media_type: mediaType,
          data: bytes.toString('base64'),
        },
      })
      continue
    }
    if (attachment.type === 'pdf') {
      const bytes = await readFile(filePath)
      content.push({
        type: 'document',
        title: attachmentName ?? undefined,
        source: {
          type: 'base64',
          media_type: 'application/pdf',
          data: bytes.toString('base64'),
        },
      })
      continue
    }
    if (attachment.type === 'text') {
      const text = await readFile(filePath, 'utf8')
      content.push({
        type: 'document',
        title: attachmentName ?? undefined,
        source: {
          type: 'text',
          media_type: 'text/plain',
          data: text,
        },
      })
    }
  }

  if (content.length === 0) {
    throw new Error('Bridge start command attachments did not produce any valid content.')
  }

  return (async function* generateUserMessage() {
    yield {
      type: 'user',
      parent_tool_use_id: null,
      message: {
        role: 'user',
        content,
      },
    }
  })()
}

function rejectAllPendingPermissions(reason) {
  for (const [requestId, pending] of pendingPermissionRequests.entries()) {
    pending.cleanup?.()
    pendingPermissionRequests.delete(requestId)
    pending.reject(new Error(reason))
  }
}

// ── Memory write round-trip ──────────────────────────────────────────────
// Mirrors the permission-request pattern: the in-process memory MCP tools
// don't write files directly. They emit `memory_write_request` over stdout,
// Swift performs the write through AgentMemoryService (single source of truth
// for size limits, staleness, atomic write), and replies with
// `memory_write_response` on stdin.
const pendingMemoryWrites = new Map()
const MEMORY_WRITE_TIMEOUT_MS = 15_000

function rejectAllPendingMemoryWrites(reason) {
  for (const [requestId, pending] of pendingMemoryWrites.entries()) {
    pending.cleanup?.()
    pendingMemoryWrites.delete(requestId)
    pending.reject(new Error(reason))
  }
}

function requestMemoryWrite(payload) {
  return new Promise((resolve, reject) => {
    const requestId = randomUUID()
    const timer = setTimeout(() => {
      if (!pendingMemoryWrites.has(requestId)) return
      pendingMemoryWrites.delete(requestId)
      reject(new Error(`Memory write request ${requestId} timed out after ${MEMORY_WRITE_TIMEOUT_MS}ms.`))
    }, MEMORY_WRITE_TIMEOUT_MS)

    pendingMemoryWrites.set(requestId, {
      resolve,
      reject,
      cleanup: () => clearTimeout(timer),
    })

    emit({ type: 'memory_write_request', requestId, ...payload })
  })
}

async function handleMemoryWriteResponse(command) {
  const requestId = normalizeString(command.requestId)
  if (!requestId) {
    emitError('memory_write_response requires requestId.', {
      code: 'missing_memory_request_id',
    })
    return
  }
  const pending = pendingMemoryWrites.get(requestId)
  if (!pending) {
    emitError(`No pending memory write found for ${requestId}.`, {
      requestId,
      code: 'unknown_memory_request',
    })
    return
  }
  pendingMemoryWrites.delete(requestId)
  pending.cleanup?.()
  if (command.ok === true) {
    pending.resolve({ ok: true, message: normalizeString(command.message) ?? 'Saved.' })
  } else {
    pending.resolve({ ok: false, message: normalizeString(command.error) ?? 'Memory write failed.' })
  }
}

// ── Skill propose round-trip ────────────────────────────────────────────
// Same shape as memory write: in-process MCP tool emits skill_write_request
// over stdout, Swift writes the SKILL.md draft to disk, replies with
// skill_write_response. Drafts go to `.cursor/skills-drafts/`, NOT
// `.cursor/skills/`, so they don't activate until the user explicitly
// promotes them.
const pendingSkillWrites = new Map()
const SKILL_WRITE_TIMEOUT_MS = 15_000

function rejectAllPendingSkillWrites(reason) {
  for (const [requestId, pending] of pendingSkillWrites.entries()) {
    pending.cleanup?.()
    pendingSkillWrites.delete(requestId)
    pending.reject(new Error(reason))
  }
}

function requestSkillWrite(payload) {
  return new Promise((resolve, reject) => {
    const requestId = randomUUID()
    const timer = setTimeout(() => {
      if (!pendingSkillWrites.has(requestId)) return
      pendingSkillWrites.delete(requestId)
      reject(new Error(`Skill write request ${requestId} timed out after ${SKILL_WRITE_TIMEOUT_MS}ms.`))
    }, SKILL_WRITE_TIMEOUT_MS)
    pendingSkillWrites.set(requestId, {
      resolve, reject,
      cleanup: () => clearTimeout(timer),
    })
    emit({ type: 'skill_write_request', requestId, ...payload })
  })
}

async function handleSkillWriteResponse(command) {
  const requestId = normalizeString(command.requestId)
  if (!requestId) {
    emitError('skill_write_response requires requestId.', { code: 'missing_skill_request_id' })
    return
  }
  const pending = pendingSkillWrites.get(requestId)
  if (!pending) {
    emitError(`No pending skill write found for ${requestId}.`, {
      requestId, code: 'unknown_skill_request',
    })
    return
  }
  pendingSkillWrites.delete(requestId)
  pending.cleanup?.()
  if (command.ok === true) {
    pending.resolve({ ok: true, message: normalizeString(command.message) ?? 'Draft saved.', path: normalizeString(command.path) })
  } else {
    pending.resolve({ ok: false, message: normalizeString(command.error) ?? 'Skill draft failed.' })
  }
}

// ── Session search round-trip ──────────────────────────────────────────
const pendingSessionSearches = new Map()
const SESSION_SEARCH_TIMEOUT_MS = 30_000

function rejectAllPendingSessionSearches(reason) {
  for (const [requestId, pending] of pendingSessionSearches.entries()) {
    pending.cleanup?.()
    pendingSessionSearches.delete(requestId)
    pending.reject(new Error(reason))
  }
}

function requestSessionSearch(payload) {
  return new Promise((resolve, reject) => {
    const requestId = randomUUID()
    const timer = setTimeout(() => {
      if (!pendingSessionSearches.has(requestId)) return
      pendingSessionSearches.delete(requestId)
      reject(new Error(`Session search ${requestId} timed out after ${SESSION_SEARCH_TIMEOUT_MS}ms.`))
    }, SESSION_SEARCH_TIMEOUT_MS)
    pendingSessionSearches.set(requestId, {
      resolve, reject,
      cleanup: () => clearTimeout(timer),
    })
    emit({ type: 'session_search_request', requestId, ...payload })
  })
}

async function handleSessionSearchResponse(command) {
  const requestId = normalizeString(command.requestId)
  if (!requestId) {
    emitError('session_search_response requires requestId.', { code: 'missing_search_request_id' })
    return
  }
  const pending = pendingSessionSearches.get(requestId)
  if (!pending) {
    emitError(`No pending session search found for ${requestId}.`, {
      requestId, code: 'unknown_search_request',
    })
    return
  }
  pendingSessionSearches.delete(requestId)
  pending.cleanup?.()
  if (command.ok === true) {
    pending.resolve({ ok: true, hits: Array.isArray(command.hits) ? command.hits : [] })
  } else {
    pending.resolve({ ok: false, message: normalizeString(command.error) ?? 'Session search failed.' })
  }
}

// ── lingcode-memory SDK MCP server ──────────────────────────────────────
// Registered when zod is available. Two tools: memory_save and memory_remove.
// Both round-trip to Swift via requestMemoryWrite().
function buildLingcodeMemoryServer() {
  if (!zodModule || !createSdkMcpServer || !sdkTool) return null
  const z = zodModule.z ?? zodModule.default?.z ?? zodModule.default
  if (!z || typeof z.object !== 'function') return null

  const VALID_TYPES = ['user', 'feedback', 'project', 'reference']
  const VALID_SCOPES = ['user', 'project']

  const saveTool = sdkTool(
    'memory_save',
    'Save or update a single memory entry. Picks the file based on `scope`: '
      + '"project" → <cwd>/.lingcode/memory.md, "user" → ~/.lingcode/USER.md. '
      + 'Use this rather than Write/Edit on memory files so the IDE can show the save and apply size limits atomically.',
    {
      scope: z.enum(VALID_SCOPES).describe('"project" for facts about this codebase, "user" for facts that travel across all projects.'),
      type: z.enum(VALID_TYPES).describe('Category tag: user (who they are), feedback (how to work), project (project context), reference (external systems).'),
      title: z.string().min(1).max(120).describe('Short section title — becomes the markdown `## <title>` header. If a section with the same title exists, it is replaced.'),
      content: z.string().min(1).max(2000).describe('The memory body. Plain markdown. Lead with the rule/fact; for feedback include a short Why; for project include a How-to-apply.'),
    },
    async (args) => {
      try {
        const result = await requestMemoryWrite({
          op: 'save',
          scope: args.scope,
          memoryType: args.type,
          title: args.title,
          content: args.content,
        })
        return {
          content: [{ type: 'text', text: result.ok ? `✓ ${result.message}` : `✗ ${result.message}` }],
          isError: !result.ok,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return { content: [{ type: 'text', text: `memory_save failed: ${message}` }], isError: true }
      }
    },
  )

  const removeTool = sdkTool(
    'memory_remove',
    'Remove a memory entry by section title from project or user memory.',
    {
      scope: z.enum(VALID_SCOPES).describe('Which memory file to edit.'),
      title: z.string().min(1).max(120).describe('Title of the section to remove. Must match exactly.'),
    },
    async (args) => {
      try {
        const result = await requestMemoryWrite({
          op: 'remove',
          scope: args.scope,
          title: args.title,
        })
        return {
          content: [{ type: 'text', text: result.ok ? `✓ ${result.message}` : `✗ ${result.message}` }],
          isError: !result.ok,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return { content: [{ type: 'text', text: `memory_remove failed: ${message}` }], isError: true }
      }
    },
  )

  const skillProposeTool = sdkTool(
    'skill_propose',
    'Propose a NEW reusable skill (slash-command) based on a procedure you just executed. Use this when you find yourself doing the same multi-step procedure twice in a session — the proposal becomes a draft SKILL.md that the user reviews before it goes live. Drafts are saved to `.cursor/skills-drafts/<name>/` (project) or `~/.cursor/skills-drafts/<name>/` (user) and do NOT activate until promoted.',
    {
      name: z.string().min(1).max(50).regex(/^[a-z0-9-]+$/).describe('Slug for the skill. Lowercase, hyphens only, e.g. "migrate-feature-flag". Becomes the directory name and slash-command (/migrate-feature-flag).'),
      scope: z.enum(VALID_SCOPES).describe('"project" if the procedure is specific to this codebase, "user" if it generalizes across all your work.'),
      description: z.string().min(10).max(200).describe('One-line summary shown when the user types `/`. Explain when to invoke the skill.'),
      body: z.string().min(50).max(8000).describe('The skill prompt — the instructions a future invocation will give to the agent. Write in second person ("you"). Be concrete: list the exact steps, name files/tools, mention non-obvious gotchas. Do not include YAML frontmatter — the IDE wraps it for you.'),
    },
    async (args) => {
      try {
        const result = await requestSkillWrite({
          op: 'propose',
          scope: args.scope,
          name: args.name,
          description: args.description,
          body: args.body,
        })
        if (result.ok) {
          const where = result.path ? ` at ${result.path}` : ''
          return {
            content: [{ type: 'text', text: `✓ Proposed skill draft '${args.name}'${where}. The user must promote it from skills-drafts/ to skills/ for it to activate.` }],
            isError: false,
          }
        } else {
          return { content: [{ type: 'text', text: `✗ ${result.message}` }], isError: true }
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return { content: [{ type: 'text', text: `skill_propose failed: ${message}` }], isError: true }
      }
    },
  )

  const skillLoadTool = sdkTool(
    'skill_load',
    'Load the full body of a skill listed in the "Available skills" index of your system prompt. The index gives only one-line descriptions; this tool returns the complete instructions for a skill so you can follow its procedure. Call it as soon as you decide a listed skill matches the task — do not paraphrase the skill before reading it.',
    {
      name: z.string().min(1).max(120).describe('Exact skill name from the index, including any plugin namespace (e.g. "superpowers:tdd").'),
    },
    async (args) => {
      const skill = availableSkills.get(args.name)
      if (!skill) {
        const known = [...availableSkills.keys()].sort().join(', ') || '(none)'
        return {
          content: [{ type: 'text', text: `Unknown skill '${args.name}'. Known skills: ${known}` }],
          isError: true,
        }
      }
      return {
        content: [{ type: 'text', text: `# Skill: ${args.name}\n\n${skill.description}\n\n${skill.body}` }],
        isError: false,
      }
    },
  )

  const sessionSearchTool = sdkTool(
    'session_search',
    'Full-text search across past Claude Agent SDK session transcripts on this Mac. Returns matching snippets with session ids and the cwd those sessions ran against. Use when the user references prior work ("the bug we hit last week", "where did we discuss X", "did I already fix Y in another project"). Index is built lazily — first call may take a few seconds while it scans transcripts.',
    {
      query: z.string().min(1).max(200).describe('FTS5 query. Plain words are AND-combined. Quote phrases with double quotes inside. Examples: terraform drift, "race condition" timer, eslint AND react.'),
      limit: z.number().int().min(1).max(20).optional().describe('Maximum number of hits to return. Default 5.'),
    },
    async (args) => {
      try {
        const result = await requestSessionSearch({
          query: args.query,
          limit: args.limit ?? 5,
        })
        if (!result.ok) {
          return { content: [{ type: 'text', text: `✗ ${result.message}` }], isError: true }
        }
        if (!result.hits || result.hits.length === 0) {
          return { content: [{ type: 'text', text: `No matches for: ${args.query}` }], isError: false }
        }
        const lines = result.hits.map((h, i) => {
          const cwd = h.cwd ? ` cwd=${h.cwd}` : ''
          const when = h.startedAt ? ` at ${h.startedAt}` : ''
          return `${i + 1}. [${h.role}]${when}${cwd}\n   session=${h.sessionId}\n   ${h.snippet}`
        })
        return { content: [{ type: 'text', text: lines.join('\n\n') }], isError: false }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return { content: [{ type: 'text', text: `session_search failed: ${message}` }], isError: true }
      }
    },
  )

  return createSdkMcpServer({
    name: 'lingcode-memory',
    version: '1.0.0',
    tools: [saveTool, removeTool, skillProposeTool, skillLoadTool, sessionSearchTool],
  })
}

const lingcodeMemoryServer = buildLingcodeMemoryServer()

// SubagentStart / SubagentStop hook callbacks. The SDK's built-in Agent tool
// dispatches subagents (often in parallel) as part of normal Task execution.
// These hooks let LingCode show live "lanes" in SubagentTreeView /
// SubagentPanelView — what's actively running, when each child started/ended.
//
// Hook return shape: an empty `{ continue: true }` is the no-op. We just want
// the side effect of emitting an event to Swift; the SDK doesn't need us to
// intervene in the agent's flow.
// Builds an SDK hooks dict that fires shell commands at the configured
// lifecycle events. The Swift host serializes its merged HooksConfig (user
// settings + project settings + plugin `hooks/hooks.json`) into the same
// {EventName: [{matcher?, hooks: [{type, command}]}]} shape used in
// .claude/settings.json. We materialize each `command` entry as an async
// callback that spawns /bin/sh with LINGCODE_*/CLAUDE_* env vars matching
// what the CLI's `runHook` provides — so the same hook script works
// identically across CLI and Mac-app/bridge sessions.
function buildShellCommandHooks(commandHooks) {
  const out = {}
  if (!commandHooks || typeof commandHooks !== 'object') return out

  for (const [eventName, eventRules] of Object.entries(commandHooks)) {
    if (!Array.isArray(eventRules)) continue
    for (const rule of eventRules) {
      const matcher = typeof rule?.matcher === 'string' && rule.matcher.length > 0
        ? rule.matcher
        : null
      const hookCmds = Array.isArray(rule?.hooks) ? rule.hooks : []
      const callbacks = []
      for (const h of hookCmds) {
        if (h && h.type === 'command' && typeof h.command === 'string' && h.command.length > 0) {
          callbacks.push(makeShellHookCallback(h.command))
        }
      }
      if (callbacks.length === 0) continue
      const entry = matcher ? { matcher, hooks: callbacks } : { hooks: callbacks }
      out[eventName] = (out[eventName] || []).concat([entry])
    }
  }
  return out
}

// Spawns `/bin/sh -c <command>` with hook env vars derived from the SDK
// callback `input`. Matches Swift's `runHook` (LingCodeAgentCore/HooksConfig.swift)
// so the same hook script works in both CLI and bridge contexts. Exit code
// 2 maps to `{ continue: false }` per Claude Code convention; everything else
// proceeds. 5-second hard timeout — hooks shouldn't be blocking workflows.
function makeShellHookCallback(shellCommand) {
  return async (input) => {
    return await new Promise((resolve) => {
      const env = { ...process.env }
      const toolName = input?.tool_name ?? ''
      const toolInput = (() => {
        try { return JSON.stringify(input?.tool_input ?? '') } catch { return '' }
      })()
      const toolResult = (() => {
        try { return JSON.stringify(input?.tool_response ?? '') } catch { return '' }
      })()
      const userPrompt = input?.prompt ?? ''
      env.LINGCODE_TOOL_NAME = toolName
      env.LINGCODE_TOOL_INPUT = toolInput
      env.LINGCODE_TOOL_RESULT = toolResult
      env.LINGCODE_USER_PROMPT = userPrompt
      env.CLAUDE_TOOL_NAME = toolName
      env.CLAUDE_TOOL_INPUT = toolInput
      env.CLAUDE_TOOL_RESULT = toolResult
      env.CLAUDE_USER_PROMPT = userPrompt

      let proc
      try {
        proc = spawn('/bin/sh', ['-c', shellCommand], {
          env,
          stdio: ['ignore', 'ignore', 'pipe'],
        })
      } catch {
        resolve({ continue: true })
        return
      }
      const timeout = setTimeout(() => {
        try { proc.kill('SIGKILL') } catch {}
      }, 5000)
      proc.once('exit', (code) => {
        clearTimeout(timeout)
        if (code === 2) {
          resolve({ continue: false, message: `hook blocked (${shellCommand.slice(0, 80)})` })
        } else {
          resolve({ continue: true })
        }
      })
      proc.once('error', () => {
        clearTimeout(timeout)
        resolve({ continue: true })
      })
    })
  }
}

// Merges two SDK hooks dicts. Used to combine the bridge's built-in subagent
// lifecycle hooks with the user/plugin shell-command hooks so neither set is
// dropped.
function mergeSDKHooks(...layers) {
  const merged = {}
  for (const layer of layers) {
    if (!layer) continue
    for (const [event, rules] of Object.entries(layer)) {
      if (!Array.isArray(rules)) continue
      merged[event] = (merged[event] || []).concat(rules)
    }
  }
  return merged
}

function buildSubagentLifecycleHooks(queryId) {
  const onStart = async (input) => {
    emit({
      type: 'subagent_started',
      queryId,
      agentId: input?.agent_id ?? null,
      agentType: input?.agent_type ?? null,
      sessionId: input?.session_id ?? null,
      cwd: input?.cwd ?? null,
    })
    return { continue: true }
  }
  const onStop = async (input) => {
    emit({
      type: 'subagent_finished',
      queryId,
      agentId: input?.agent_id ?? null,
      agentType: input?.agent_type ?? null,
      sessionId: input?.session_id ?? null,
      transcriptPath: input?.agent_transcript_path ?? null,
      stopHookActive: input?.stop_hook_active ?? null,
    })
    return { continue: true }
  }
  return {
    SubagentStart: [{ hooks: [onStart] }],
    SubagentStop: [{ hooks: [onStop] }],
  }
}

function createPermissionHandler(queryId, permissionMode) {
  return async (toolName, input, options = {}) =>
    permissionMode === 'dontAsk'
      ? {
          behavior: 'deny',
          message:
            options.decisionReason ??
            'LingCode permission mode is set to Don’t Ask.',
          decisionClassification: 'user_reject',
        }
      :
    new Promise((resolve, reject) => {
      const requestId = randomUUID()
      let settled = false

      const cleanup = () => {
        settled = true
        if (options.signal && abortHandler) {
          options.signal.removeEventListener('abort', abortHandler)
        }
      }

      const abortHandler = () => {
        if (settled) return
        cleanup()
        pendingPermissionRequests.delete(requestId)
        reject(new Error(`Permission request ${requestId} aborted.`))
      }

      if (options.signal?.aborted) {
        abortHandler()
        return
      }

      if (options.signal) {
        options.signal.addEventListener('abort', abortHandler, { once: true })
      }

      pendingPermissionRequests.set(requestId, {
        resolve: (result) => {
          if (settled) return
          cleanup()
          resolve(result)
        },
        reject: (error) => {
          if (settled) return
          cleanup()
          reject(error)
        },
        cleanup,
        originalInput: input,
      })

      emit({
        type: 'permission_request',
        queryId,
        requestId,
        toolName,
        input,
        options: {
          suggestions: options.suggestions ?? [],
          blockedPath: options.blockedPath ?? null,
          decisionReason: options.decisionReason ?? null,
          title: options.title ?? null,
          displayName: options.displayName ?? null,
          description: options.description ?? null,
          toolUseID: options.toolUseID,
          agentID: options.agentID ?? null,
        },
      })
    })
}

async function runPrompt(command) {
  if (activeQuery) {
    emitError('A Claude query is already running.', {
      queryId: activeQueryId,
      code: 'query_already_running',
    })
    return
  }

  let prompt
  try {
    prompt = await buildPromptInput(command)
  } catch (error) {
    emitError(error instanceof Error ? error.message : String(error), {
      code: 'missing_prompt',
    })
    return
  }

  const queryId = normalizeString(command.queryId) ?? randomUUID()
  const permissionMode =
    normalizePermissionMode(command.permissionMode) ?? defaultPermissionMode
  const cwd = normalizeString(command.cwd) ?? process.cwd()
  const claudePath = normalizeString(command.claudePath) ?? undefined
  const resumeSessionId =
    normalizeString(command.resumeSessionId) ?? currentSessionId ?? undefined
  const maxTurns =
    typeof command.maxTurns === 'number' && command.maxTurns > 0
      ? Math.floor(command.maxTurns)
      : 50
  const allowedTools = Array.isArray(command.allowedTools) ? command.allowedTools : undefined
  const disallowedTools = Array.isArray(command.disallowedTools) ? command.disallowedTools : undefined
  // Build mcpServers as an object map (SDK form). Swift sends `{name: cfg}`;
  // we also fold in the in-process `lingcode-memory` SDK server when available.
  const mcpServers = (() => {
    const out = {}
    if (command.mcpServers && typeof command.mcpServers === 'object') {
      for (const [name, cfg] of Object.entries(command.mcpServers)) {
        if (cfg && typeof cfg === 'object') out[name] = cfg
      }
    }
    if (lingcodeMemoryServer) {
      out['lingcode-memory'] = lingcodeMemoryServer
    }
    return Object.keys(out).length > 0 ? out : undefined
  })()
  const customSystemPrompt = normalizeString(command.systemPrompt)
  const customAppendSystemPrompt = normalizeString(command.appendSystemPrompt)
  // Custom subagent definitions, parsed Swift-side from `.claude/agents/<name>.md`.
  // SDK expects `agents: Record<string, {description, prompt, tools?, model?}>`.
  const customAgents = command.agents && typeof command.agents === 'object' && !Array.isArray(command.agents)
    ? Object.fromEntries(
        Object.entries(command.agents)
          .filter(([k, v]) => typeof k === 'string' && v && typeof v === 'object' && typeof v.description === 'string' && typeof v.prompt === 'string')
          .map(([k, v]) => {
            const def = { description: v.description, prompt: v.prompt }
            if (Array.isArray(v.tools) && v.tools.every(t => typeof t === 'string')) def.tools = v.tools
            if (typeof v.model === 'string' && v.model.length > 0) def.model = v.model
            return [k, def]
          })
      )
    : undefined
  const additionalDirectories = Array.isArray(command.additionalDirectories) && command.additionalDirectories.length
    ? command.additionalDirectories.filter((d) => typeof d === 'string' && d.length > 0)
    : undefined
  const thinkingEnabled = command.thinking === true

  // Refresh the per-query skills registry. The Swift host walks `.claude/skills/`
  // and Claude Code-format plugin trees, then ships the result as command.skills.
  // We rebuild the index string here so a freshly installed skill shows up on
  // the next prompt without restarting the bridge.
  availableSkills.clear()
  if (Array.isArray(command.skills)) {
    for (const s of command.skills) {
      if (s && typeof s.name === 'string' && typeof s.description === 'string' && typeof s.body === 'string') {
        availableSkills.set(s.name, { description: s.description, body: s.body })
      }
    }
  }
  const skillIndex = availableSkills.size > 0
    ? [
        '## Available skills',
        '',
        'These pre-authored procedures are available for this session. When a task matches a skill, call the `skill_load` tool with the exact name to read its full body before acting — paraphrasing from the description alone is not enough.',
        '',
        ...[...availableSkills.entries()]
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([name, s]) => `- \`${name}\` — ${s.description}`),
      ].join('\n')
    : null

  defaultPermissionMode = permissionMode
  activeAbortController = new AbortController()

  // Defense in depth: re-apply provider env in case anything between the last
  // set_model and now mutated it. The Anthropic SDK spawns the `claude` CLI
  // fresh on every query() call, so env mutation here propagates to the child.
  applyProviderEnv(currentModel)
  // Resolve the wire model id we send to the SDK:
  //   `lingbackend:<uuid>:<modelId>` → unwrap to the upstream's model id (deepseek-v4-flash, glm-4.6, …)
  //   LingModel tier sentinel → upstream tier id (auto / auto-advanced)
  //   anything else → as-is (Anthropic native: claude-opus-4-7 etc.)
  const effectiveModel = (() => {
    const byok = parseBackendModelTag(currentModel)
    if (byok) return byok.modelId
    if (isLingModelTag(currentModel)) return lingModelUpstream(currentModel)
    return currentModel
  })()

  emit({
    type: 'query_started',
    queryId,
    cwd,
    permissionMode,
    resumeSessionId: resumeSessionId ?? null,
  })

  const queryOptions = {
    cwd,
    maxTurns,
    includePartialMessages: true,
    permissionMode,
    abortController: activeAbortController,
    ...(permissionMode === 'bypassPermissions'
      ? { allowDangerouslySkipPermissions: true }
      : {}),
    ...(claudePath ? { pathToClaudeCodeExecutable: claudePath } : {}),
    ...(resumeSessionId ? { resume: resumeSessionId } : {}),
    ...(effectiveModel ? { model: effectiveModel } : {}),
    ...(allowedTools ? { allowedTools } : {}),
    ...(disallowedTools ? { disallowedTools } : {}),
    ...(mcpServers ? { mcpServers } : {}),
    ...(customSystemPrompt ? { systemPrompt: customSystemPrompt } : {}),
    ...(additionalDirectories ? { additionalDirectories } : {}),
    ...(thinkingEnabled ? { thinking: { type: 'enabled', budget_tokens: 8000 } } : {}),
    ...(customAgents && Object.keys(customAgents).length > 0 ? { agents: customAgents } : {}),
    appendSystemPrompt: [
      isLingModelTag(currentModel) ? LINGMODEL_IDENTITY_DIRECTIVE : null,
      NARRATION_DIRECTIVE,
      skillIndex,
      customAppendSystemPrompt || null,
    ].filter(Boolean).join('\n\n'),
    canUseTool: createPermissionHandler(queryId, permissionMode),
    hooks: mergeSDKHooks(
      buildSubagentLifecycleHooks(queryId),
      buildShellCommandHooks(command.hooks),
    ),
    agentProgressSummaries: true,
  }

  const stream = query({ prompt, options: queryOptions })
  activeQuery = stream
  activeQueryId = queryId

  let finalResult = null

  try {
    for await (const message of stream) {
      if (message && typeof message === 'object') {
        if (typeof message.session_id === 'string' && message.session_id) {
          currentSessionId = message.session_id
        }
        if (message.type === 'result') {
          finalResult = message
        }
        if (message.type === 'rate_limit_event') {
          emit({
            type: 'rate_limit_event',
            queryId,
            rate_limit_info: message.rate_limit_info,
          })
          continue
        }
      }
      emit({ type: 'sdk_message', queryId, message })
    }

    emit({
      type: 'query_finished',
      queryId,
      sessionId: currentSessionId,
      result: finalResult,
    })
  } catch (error) {
    if (activeAbortController?.signal.aborted) {
      emit({
        type: 'query_cancelled',
        queryId,
        sessionId: currentSessionId,
        message: error instanceof Error ? error.message : 'Query cancelled.',
      })
      return
    }
    emit({
      type: 'query_failed',
      queryId,
      sessionId: currentSessionId,
      message: error instanceof Error ? error.message : String(error),
    })
  } finally {
    activeQuery = null
    activeQueryId = null
    activeAbortController = null
    rejectAllPendingPermissions('Query finished before permission response arrived.')
  }
}

async function handlePermissionResponse(command) {
  const requestId = normalizeString(command.requestId)
  if (!requestId) {
    emitError('permission_response requires requestId.', {
      code: 'missing_permission_request_id',
    })
    return
  }

  const pending = pendingPermissionRequests.get(requestId)
  if (!pending) {
    emitError(`No pending permission request found for ${requestId}.`, {
      requestId,
      code: 'unknown_permission_request',
    })
    return
  }
  pendingPermissionRequests.delete(requestId)

  const behavior = command.behavior === 'deny' ? 'deny' : 'allow'
  if (behavior === 'allow') {
    const resolvedInput = command.updatedInput ?? pending.originalInput ?? {}
    pending.resolve({
      behavior: 'allow',
      updatedInput: resolvedInput,
      ...(Array.isArray(command.updatedPermissions)
        ? { updatedPermissions: command.updatedPermissions }
        : {}),
      ...(normalizeString(command.decisionClassification)
        ? { decisionClassification: command.decisionClassification }
        : {}),
    })
  } else {
    pending.resolve({
      behavior: 'deny',
      message:
        normalizeString(command.message) ??
        'Denied by LingCode bridge approval UI.',
      ...(typeof command.interrupt === 'boolean'
        ? { interrupt: command.interrupt }
        : {}),
      ...(normalizeString(command.decisionClassification)
        ? { decisionClassification: command.decisionClassification }
        : {}),
    })
  }

  emit({
    type: 'permission_resolved',
    requestId,
    behavior,
  })
}

async function handleCommand(command) {
  if (!command || typeof command !== 'object') {
    emitError('Bridge received a non-object command payload.', {
      code: 'invalid_command_shape',
    })
    return
  }

  switch (command.type) {
    case 'start':
      await runPrompt(command)
      break
    case 'mock_can_use_tool_roundtrip':
      await runMockCanUseToolRoundTrip(command)
      break
    case 'mock_pending_permission':
      await runMockPendingPermission(command)
      break
    case 'permission_response':
      await handlePermissionResponse(command)
      break
    case 'memory_write_response':
      await handleMemoryWriteResponse(command)
      break
    case 'skill_write_response':
      await handleSkillWriteResponse(command)
      break
    case 'session_search_response':
      await handleSessionSearchResponse(command)
      break
    case 'reset_session':
      currentSessionId = null
      emit({ type: 'session_reset' })
      break
    case 'set_permission_mode': {
      const next = normalizePermissionMode(command.permissionMode)
      if (!next) {
        emitError('Unknown permission mode.', {
          code: 'invalid_permission_mode',
          permissionMode: command.permissionMode ?? null,
        })
        return
      }
      defaultPermissionMode = next
      emit({ type: 'permission_mode_updated', permissionMode: next })
      break
    }
    case 'set_model': {
      currentModel = normalizeString(command.model)
      const newToken = normalizeString(command.proxyAuthToken)
      if (newToken) proxyAuthToken = newToken
      applyProviderEnv(currentModel)
      emit({ type: 'model_updated', model: currentModel })
      break
    }
    case 'set_backends': {
      // Sent by Swift on bridge spawn AND any time the user adds/edits/
      // deletes a BYOK backend in Settings → Claude Backends. The table
      // is consulted by applyProviderEnv() when a `lingbackend:<uuid>:…`
      // model tag is selected.
      const list = Array.isArray(command.backends) ? command.backends : []
      backendsByUuid.clear()
      let added = 0
      for (const b of list) {
        if (b && typeof b.uuid === 'string' && b.uuid) {
          backendsByUuid.set(b.uuid, b)
          added += 1
        }
      }
      // Re-apply env in case the currently-selected model is a backend
      // tag and its config just changed (key rotated, URL edited).
      applyProviderEnv(currentModel)
      emit({ type: 'backends_updated', count: added })
      break
    }
    case 'cancel_active_query':
      if (!activeQuery || !activeAbortController) {
        emit({
          type: 'query_cancelled',
          queryId: activeQueryId,
          sessionId: currentSessionId,
          message: 'No active Claude query to cancel.',
          alreadyIdle: true,
        })
        return
      }
      activeAbortController.abort(new Error('Cancelled by LingCode.'))
      emit({
        type: 'cancel_requested',
        queryId: activeQueryId,
        sessionId: currentSessionId,
      })
      break
    case 'ping':
      emit({
        type: 'pong',
        activeQueryId,
        sessionId: currentSessionId,
      })
      break
    case 'shutdown':
      activeAbortController?.abort(new Error('Bridge shutting down.'))
      rejectAllPendingPermissions('Bridge shutting down.')
      emit({ type: 'shutdown_ack' })
      process.exit(0)
      break
    default:
      emitError(`Unknown bridge command type: ${String(command.type)}`, {
        code: 'unknown_command_type',
      })
      break
  }
}

async function runMockCanUseToolRoundTrip(command) {
  if (activeQuery) {
    emitError('A Claude query is already running.', {
      queryId: activeQueryId,
      code: 'query_already_running',
    })
    return
  }

  const queryId = normalizeString(command.queryId) ?? randomUUID()
  const targetPath = normalizeString(command.targetPath)
  if (!targetPath) {
    emitError('mock_can_use_tool_roundtrip requires targetPath.', {
      code: 'missing_target_path',
    })
    return
  }

  emit({
    type: 'query_started',
    queryId,
    cwd: normalizeString(command.cwd) ?? process.cwd(),
    permissionMode: 'default',
    resumeSessionId: null,
    mock: true,
  })

  activeQuery = { mock: true }
  activeQueryId = queryId

  try {
    const decide = createPermissionHandler(queryId)
    const result = await decide(
      'Edit',
      {
        file_path: targetPath,
        old_string: '',
        new_string: 'APPROVED_ROUND_TRIP\n',
      },
      {
        toolUseID: `mock-tool-${queryId}`,
        title: 'Claude wants to edit a file',
        displayName: 'Edit file',
        description: `Write APPROVED_ROUND_TRIP to ${targetPath}`,
        blockedPath: targetPath,
        decisionReason: 'Bridge mock self-test for canUseTool round-trip.',
        suggestions: [
          {
            type: 'addDirectories',
            directories: [targetPath],
            destination: 'session',
          },
        ],
      }
    )

    if (result.behavior === 'deny') {
      emit({
        type: 'query_failed',
        queryId,
        sessionId: currentSessionId,
        message: result.message,
      })
      return
    }

    await writeFile(targetPath, 'APPROVED_ROUND_TRIP\n', 'utf8')

    emit({
      type: 'sdk_message',
      queryId,
      message: {
        type: 'assistant',
        message: {
          content: [
            {
              type: 'text',
              text: 'Mock bridge self-test completed after an approved tool round-trip.',
            },
          ],
        },
        parent_tool_use_id: null,
        session_id: currentSessionId ?? `mock-session-${queryId}`,
      },
    })

    emit({
      type: 'query_finished',
      queryId,
      sessionId: currentSessionId ?? `mock-session-${queryId}`,
      result: {
        type: 'result',
        subtype: 'success',
        is_error: false,
        result: 'Mock round-trip completed.',
        session_id: currentSessionId ?? `mock-session-${queryId}`,
      },
    })
  } catch (error) {
    emit({
      type: 'query_failed',
      queryId,
      sessionId: currentSessionId,
      message: error instanceof Error ? error.message : String(error),
    })
  } finally {
    activeQuery = null
    activeQueryId = null
    rejectAllPendingPermissions('Mock query finished before permission response arrived.')
  }
}

async function runMockPendingPermission(command) {
  if (activeQuery) {
    emitError('A Claude query is already running.', {
      queryId: activeQueryId,
      code: 'query_already_running',
    })
    return
  }

  const queryId = normalizeString(command.queryId) ?? randomUUID()
  activeAbortController = new AbortController()
  activeQuery = { mock: true, pendingPermission: true }
  activeQueryId = queryId

  emit({
    type: 'query_started',
    queryId,
    cwd: normalizeString(command.cwd) ?? process.cwd(),
    permissionMode: 'default',
    resumeSessionId: null,
    mock: true,
  })

  try {
    const decide = createPermissionHandler(queryId, 'default')
    await decide(
      'Edit',
      { file_path: '/tmp/mock-cancel.txt' },
      {
        toolUseID: `mock-cancel-${queryId}`,
        title: 'Claude wants to edit a file',
        displayName: 'Edit file',
        description: 'Waiting for approval or cancellation.',
        blockedPath: '/tmp/mock-cancel.txt',
        decisionReason: 'Bridge mock cancellation test.',
        signal: activeAbortController.signal,
      }
    )

    emit({
      type: 'query_finished',
      queryId,
      sessionId: currentSessionId ?? `mock-session-${queryId}`,
      result: {
        type: 'result',
        subtype: 'success',
        is_error: false,
        result: 'Mock pending permission resolved.',
        session_id: currentSessionId ?? `mock-session-${queryId}`,
      },
    })
  } catch (error) {
    if (activeAbortController?.signal.aborted) {
      emit({
        type: 'query_cancelled',
        queryId,
        sessionId: currentSessionId,
        message: error instanceof Error ? error.message : 'Query cancelled.',
      })
      return
    }

    emit({
      type: 'query_failed',
      queryId,
      sessionId: currentSessionId,
      message: error instanceof Error ? error.message : String(error),
    })
  } finally {
    activeQuery = null
    activeQueryId = null
    activeAbortController = null
    rejectAllPendingPermissions('Mock pending permission finished before permission response arrived.')
  }
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
})

rl.on('line', async (line) => {
  const trimmed = line.trim()
  if (!trimmed) return

  let command
  try {
    command = JSON.parse(trimmed)
  } catch (error) {
    emitError('Bridge received invalid JSON.', {
      code: 'invalid_json',
      detail: error instanceof Error ? error.message : String(error),
    })
    return
  }

  try {
    await handleCommand(command)
  } catch (error) {
    emitError(error instanceof Error ? error.message : String(error), {
      code: 'bridge_command_failed',
    })
  }
})

rl.on('close', () => {
  rejectAllPendingPermissions('Bridge stdin closed.')
  rejectAllPendingMemoryWrites('Bridge stdin closed.')
  rejectAllPendingSkillWrites('Bridge stdin closed.')
  rejectAllPendingSessionSearches('Bridge stdin closed.')
  process.exit(0)
})

process.on('SIGINT', () => {
  rejectAllPendingPermissions('Bridge interrupted.')
  rejectAllPendingMemoryWrites('Bridge interrupted.')
  rejectAllPendingSkillWrites('Bridge interrupted.')
  rejectAllPendingSessionSearches('Bridge interrupted.')
  process.exit(130)
})

process.on('SIGTERM', () => {
  rejectAllPendingPermissions('Bridge terminated.')
  rejectAllPendingMemoryWrites('Bridge terminated.')
  rejectAllPendingSkillWrites('Bridge terminated.')
  rejectAllPendingSessionSearches('Bridge terminated.')
  process.exit(143)
})

emit({
  type: 'ready',
  pid: process.pid,
  protocolVersion: 1,
  permissionModes: [...VALID_PERMISSION_MODES],
})
