import { addMemories, retrieveMemories, type Mem0ConfigSettings } from '@mem0/vercel-ai-provider';
import { ChatAnthropic } from '@langchain/anthropic';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { ChatOpenAI } from '@langchain/openai';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  isAIMessage,
} from '@langchain/core/messages';
import { DynamicStructuredTool } from '@langchain/core/tools';
import type { Channel, Event, MessageResponse, StreamChat } from 'stream-chat';
import { z, type ZodTypeAny } from 'zod';
import type { AIAgent } from './types';
import { AgentPlatform } from './types';

type ChatXAIConstructor = new (fields: ConstructorParameters<typeof ChatOpenAI>[0]) => BaseChatModel;

const loadChatXAIConstructor = (): ChatXAIConstructor | undefined => {
  const moduleIds = [
    '@langchain/xai',
    '@langchain/community/chat_models/xai',
  ];
  for (const moduleId of moduleIds) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires, global-require
      const mod = require(moduleId);
      if (mod?.ChatXAI) {
        return mod.ChatXAI as ChatXAIConstructor;
      }
    } catch {
      // Optional dependency missing; continue to the next candidate.
    }
  }
  return undefined;
};

const ChatXAIClass = loadChatXAIConstructor();
const XAI_DEFAULT_BASE_URL = 'https://api.x.ai/v1';

const createXaiModel = (apiKey: string, modelId?: string): BaseChatModel => {
  const resolvedModel = modelId ?? 'grok-beta';
  if (ChatXAIClass) {
    return new ChatXAIClass({
      apiKey,
      streaming: true,
      model: resolvedModel,
    });
  }
  const baseURL = process.env.XAI_BASE_URL?.trim() || XAI_DEFAULT_BASE_URL;
  return new ChatOpenAI({
    apiKey,
    streaming: true,
    modelName: resolvedModel,
    configuration: {
      baseURL,
    },
  });
};

const BASE_SYSTEM_PROMPT =
  'You are an AI assistant. Help users with their questions. Evaluate each user turn independently: restate the user intent, decide whether it matches any available tool instructions, and only invoke the matching tool when the intent clearly applies. If no tool matches, answer normally.';
const CLIENT_TOOL_EVENT = 'custom_client_tool_invocation';

const DEFAULT_MEM0_AGENT_ID =
  process.env.MEM0_DEFAULT_AGENT_ID?.trim() || 'stream-ai-agent';
const DEFAULT_MEM0_APP_ID =
  process.env.MEM0_DEFAULT_APP_ID?.trim() || 'stream-ai-app';

const MEM0_CONFIG_FROM_ENV: Mem0ConfigSettings | undefined = (() => {
  const raw = process.env.MEM0_CONFIG_JSON;
  if (!raw) return undefined;
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Mem0ConfigSettings;
    }
    console.warn('MEM0_CONFIG_JSON must be a JSON object. Ignoring value.');
  } catch (error) {
    console.warn('Failed to parse MEM0_CONFIG_JSON. Ignoring value.', error);
  }
  return undefined;
})();

type IndicatorState =
  | 'AI_STATE_THINKING'
  | 'AI_STATE_GENERATING'
  | 'AI_STATE_EXTERNAL_SOURCES'
  | 'AI_STATE_ERROR';

type CoreMessage = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

type LangChainTool = DynamicStructuredTool;

type ToolCall = {
  name: string;
  args: unknown;
  id?: string;
};

export interface Mem0ContextInput {
  userId?: string;
  channelId?: string;
  agentId?: string;
  appId?: string;
  configOverrides?: Partial<Mem0ConfigSettings>;
}

interface CreateModelOptions {
  disableMem0?: boolean;
}

export interface ToolExecutionContext {
  channel: Channel;
  message: MessageResponse;
  sendEvent: (event: Record<string, unknown>) => Promise<void>;
}

export interface AgentTool {
  name: string;
  description: string;
  instructions?: string;
  parameters: ZodTypeAny;
  execute: (args: unknown, context: ToolExecutionContext) => Promise<string> | string;
  showExternalSourcesIndicator?: boolean;
}

export interface JsonSchemaDefinition {
  type?: string;
  description?: string;
  enum?: Array<string | number | boolean>;
  properties?: Record<string, JsonSchemaDefinition>;
  items?: JsonSchemaDefinition | JsonSchemaDefinition[];
  required?: string[];
  additionalProperties?: boolean | JsonSchemaDefinition;
  [key: string]: unknown;
}

export interface ClientToolDefinition {
  name: string;
  description: string;
  instructions?: string;
  parameters?: JsonSchemaDefinition;
  showExternalSourcesIndicator?: boolean;
}

const buildMem0Settings = (context?: Mem0ContextInput): Mem0ConfigSettings => {
  const envDefaults = MEM0_CONFIG_FROM_ENV ? { ...MEM0_CONFIG_FROM_ENV } : {};
  const overrides = context?.configOverrides ? { ...context.configOverrides } : {};
  const merged: Mem0ConfigSettings = {
    ...envDefaults,
    ...overrides,
  };

  const envUserId = process.env.MEM0_DEFAULT_USER_ID?.trim();
  const envAgentId = process.env.MEM0_DEFAULT_AGENT_ID?.trim();
  const envAppId = process.env.MEM0_DEFAULT_APP_ID?.trim();
  const fallbackChannel = context?.channelId;

  merged.user_id =
    context?.userId ??
    overrides.user_id ??
    merged.user_id ??
    envUserId ??
    fallbackChannel ??
    context?.agentId;

  merged.agent_id =
    context?.agentId ??
    overrides.agent_id ??
    merged.agent_id ??
    envAgentId ??
    context?.userId;

  merged.app_id =
    context?.appId ??
    overrides.app_id ??
    merged.app_id ??
    envAppId ??
    fallbackChannel;

  return merged;
};

const cloneMem0Context = (
  context?: Mem0ContextInput,
): Mem0ContextInput | undefined => {
  if (!context) {
    return undefined;
  }
  const overrides = context.configOverrides
    ? { ...context.configOverrides }
    : undefined;
  if (overrides?.metadata) {
    overrides.metadata = { ...overrides.metadata };
  }
  return {
    ...context,
    configOverrides: overrides,
  };
};

const toMem0Prompt = (messages: CoreMessage[]): Array<{ role: string; content: string }> =>
  messages.map((msg) => ({ role: msg.role, content: msg.content }));

export const createModelForPlatform = (
  platform: AgentPlatform,
  modelOverride?: string,
  _options?: CreateModelOptions,
): BaseChatModel => {
  const modelId =
    typeof modelOverride === 'string' && modelOverride.trim()
      ? modelOverride.trim()
      : undefined;

  switch (platform) {
    case AgentPlatform.OPENAI: {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error('OpenAI API key is required');
      }
      return new ChatOpenAI({
        apiKey,
        streaming: true,
        modelName: modelId ?? 'gpt-4o-mini',
      });
    }
    case AgentPlatform.ANTHROPIC: {
      const apiKey = process.env.ANTHROPIC_API_KEY;
      if (!apiKey) {
        throw new Error('Anthropic API key is required');
      }
      return new ChatAnthropic({
        anthropicApiKey: apiKey,
        streaming: true,
        model: modelId ?? 'claude-3-5-sonnet-20241022',
      });
    }
    case AgentPlatform.GEMINI: {
      const apiKey =
        process.env.GOOGLE_GENERATIVE_AI_API_KEY ?? process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error('Gemini API key is required');
      }
      return new ChatGoogleGenerativeAI({
        apiKey,
        streaming: true,
        model: modelId ?? 'gemini-1.5-flash',
      });
    }
    case AgentPlatform.XAI: {
      const apiKey = process.env.XAI_API_KEY;
      if (!apiKey) {
        throw new Error('xAI API key is required');
      }
      return createXaiModel(apiKey, modelId);
    }
    default:
      throw new Error(`Unsupported AI platform: ${platform}`);
  }
};

export class LangChainAIAgent implements AIAgent {
  private lastInteractionTs = Date.now();
  private handlers = new Set<LangChainResponseHandler>();
  private serverTools: AgentTool[];
  private clientTools: AgentTool[] = [];
  private readonly modelOverride?: string;
  private readonly additionalInstructions: string[];
  private readonly mem0Context?: Mem0ContextInput;

  constructor(
    readonly chatClient: StreamChat,
    readonly channel: Channel,
    private readonly platform: AgentPlatform,
    tools: AgentTool[] = [],
    modelOverride?: string,
    additionalInstructions?: string[],
    mem0Context?: Mem0ContextInput,
  ) {
    this.serverTools = tools ?? [];
    this.modelOverride = modelOverride;
    this.additionalInstructions = Array.isArray(additionalInstructions)
      ? additionalInstructions.filter((line) => line && line.trim().length)
      : additionalInstructions
        ? [additionalInstructions]
        : [];
    this.mem0Context = mem0Context;
  }

  init = async () => {
    this.createLanguageModel();
    this.chatClient.on('message.new', this.handleMessage);
  };

  dispose = async () => {
    this.chatClient.off('message.new', this.handleMessage);
    await this.chatClient.disconnectUser();

    const handlers = Array.from(this.handlers);
    this.handlers.clear();
    await Promise.allSettled(handlers.map((handler) => handler.dispose()));
  };

  getLastInteraction = (): number => this.lastInteractionTs;

  setServerTools = (tools: AgentTool[]) => {
    this.serverTools = tools ?? [];
  };

  addServerTools = (tools: AgentTool[]) => {
    this.serverTools = [...this.serverTools, ...(tools ?? [])];
  };

  setClientTools = (tools: AgentTool[]) => {
    this.clientTools = tools ?? [];
  };

  setClientToolDefinitions = (definitions: ClientToolDefinition[]) => {
    const tools = (definitions ?? []).map((definition) =>
      this.createClientTool(definition),
    );
    this.setClientTools(tools);
  };

  private createLanguageModel(userId?: string): BaseChatModel {
    return createModelForPlatform(this.platform, this.modelOverride, {
      disableMem0: !this.mem0Context && !userId,
    });
  }

  private buildMem0ContextForUser(
    userId?: string,
  ): Mem0ContextInput | undefined {
    let base = cloneMem0Context(this.mem0Context);
    if (!base && this.channel?.id) {
      base = {
        channelId: this.channel.id,
      };
    }
    if (!base && !userId) {
      return undefined;
    }
    const context: Mem0ContextInput = base ?? {};
    if (!context.agentId) {
      context.agentId = DEFAULT_MEM0_AGENT_ID;
    }
    if (!context.appId) {
      context.appId = DEFAULT_MEM0_APP_ID;
    }
    if (!context.channelId && this.channel?.id) {
      context.channelId = this.channel.id;
    }
    if (userId) {
      context.userId = userId;
      const metadata: Record<string, any> = {
        ...(context.configOverrides?.metadata ?? {}),
        user_id: userId,
      };
      context.configOverrides = {
        ...(context.configOverrides ?? {}),
        metadata,
      };
    }
    return context;
  }

  private getActiveTools(): AgentTool[] {
    return [...this.serverTools, ...this.clientTools];
  }

  private createClientTool(definition: ClientToolDefinition): AgentTool {
    const parameters =
      jsonSchemaToZod(definition.parameters) ?? z.object({}).passthrough();
    return {
      name: definition.name,
      description: definition.description,
      instructions: definition.instructions,
      parameters,
      showExternalSourcesIndicator: definition.showExternalSourcesIndicator,
      execute: async (args, context) => {
        console.log(
          `[ClientTool] Dispatching ${definition.name} with args:`,
          args,
        );
        await context.sendEvent({
          type: CLIENT_TOOL_EVENT,
          cid: context.message.cid,
          message_id: context.message.id,
          channel_id: context.channel.id,
          channel_type: context.channel.type,
          tool: {
            name: definition.name,
            description: definition.description,
            instructions: definition.instructions,
            parameters: definition.parameters ?? null,
          },
          args: args ?? {},
        });
        return `Client tool "${definition.name}" invocation dispatched.`;
      },
    };
  }

  private getSystemPrompt(): string {
    const instructions = [
      ...this.additionalInstructions,
      ...this.getActiveTools()
        .map((tool) => tool.instructions ?? '')
        .filter((value) => typeof value === 'string' && value.trim().length > 0),
    ]
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    if (!instructions.length) {
      return BASE_SYSTEM_PROMPT;
    }

    const formatted = instructions.map((line) => `- ${line}`).join('\n');
    return `${BASE_SYSTEM_PROMPT}\n\nGuidelines:\n${formatted}`;
  }

  private async getMem0SystemPrompt(
    history: CoreMessage[],
    userId?: string,
  ): Promise<string | null> {
    if (!this.mem0Context) {
      return null;
    }
    const context = this.buildMem0ContextForUser(userId);
    if (!context) {
      return null;
    }
    try {
      const mem0Prompt = await retrieveMemories(toMem0Prompt(history), {
        ...buildMem0Settings(context),
      });
      if (mem0Prompt && mem0Prompt.trim().length) {
        return mem0Prompt.trim();
      }
    } catch (error) {
      console.warn('Mem0 retrieval failed', error);
    }
    return null;
  }

  private async persistMem0Messages(
    history: CoreMessage[],
    assistantText: string,
    userId?: string,
  ): Promise<void> {
    if (!this.mem0Context) return;
    const context = this.buildMem0ContextForUser(userId);
    if (!context) return;
    try {
      const prompt = toMem0Prompt([
        ...history,
        { role: 'assistant', content: assistantText },
      ]);
      await addMemories(prompt, buildMem0Settings(context));
    } catch (error) {
      console.warn('Mem0 update failed', error);
    }
  }

  private handleMessage = async (event: Event) => {
    const incomingMessage = event.message as MessageResponse & {
      ai_generated?: boolean;
    };
    if (!incomingMessage || incomingMessage.ai_generated) {
      return;
    }

    const incomingText = incomingMessage.text?.trim();
    if (!incomingText) {
      return;
    }

    const userId = incomingMessage.user?.id;
    if (!userId || userId.startsWith('ai-bot')) return;

    this.lastInteractionTs = Date.now();

    const history = this.channel.state.messages
      .slice(-10)
      .filter((msg) => msg.text && msg.text.trim() !== '')
      .map<CoreMessage>((msg) => ({
        role: msg.user?.id.startsWith('ai-bot') ? 'assistant' : 'user',
        content: msg.text ?? '',
      }));

    const lastHistoryEntry = history[history.length - 1];
    if (
      lastHistoryEntry?.content !== incomingText ||
      lastHistoryEntry.role !== 'user'
    ) {
      history.push({ role: 'user', content: incomingText });
    }

    const systemPrompt = this.getSystemPrompt();
    const mem0Prompt = await this.getMem0SystemPrompt(history, userId);

    const messages: CoreMessage[] = [
      { role: 'system', content: systemPrompt },
      ...(mem0Prompt ? [{ role: 'system', content: mem0Prompt }] : []),
      ...history,
    ];

    const { message: channelMessage } = await this.channel.sendMessage({
      text: '',
      ai_generated: true,
    } as any);

    await this.safeSendEvent({
      type: 'ai_indicator.update',
      ai_state: 'AI_STATE_THINKING',
      cid: channelMessage.cid,
      message_id: channelMessage.id,
    });

    let languageModel: BaseChatModel;
    try {
      languageModel = this.createLanguageModel(userId);
    } catch (error) {
      console.error('Failed to initialize language model', error);
      await this.handlePreflightError(channelMessage, error as Error);
      return;
    }

    const handler = new LangChainResponseHandler(
      languageModel,
      this.chatClient,
      this.channel,
      channelMessage,
      messages,
      (event) => this.safeSendEvent(event),
      () => this.getActiveTools(),
      (responseText) => this.persistMem0Messages(history, responseText, userId),
    );

    this.handlers.add(handler);
    void handler
      .run()
      .catch((error) => {
        console.error('AI handler error', error);
      })
      .finally(() => {
        this.handlers.delete(handler);
      });
  };

  private async handlePreflightError(
    message: MessageResponse,
    error: Error,
  ): Promise<void> {
    await this.safeSendEvent({
      type: 'ai_indicator.update',
      ai_state: 'AI_STATE_ERROR',
      cid: message.cid,
      message_id: message.id,
    });
    await this.chatClient.partialUpdateMessage(message.id, {
      set: {
        text: error.message ?? 'Error generating the message',
        generating: false,
      } as any,
    });
  }

  private async safeSendEvent(event: Record<string, unknown>) {
    const maxAttempts = 5;
    let delay = 100;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        await this.channel.sendEvent(event as any);
        return;
      } catch (err) {
        const status = (err as any)?.status || (err as any)?.response?.status;
        const retryable = status === 429 || (status >= 500 && status < 600);
        if (!retryable || attempt === maxAttempts) {
          const label = retryable
            ? 'Failed to send event after retries'
            : 'Failed to send event';
          console.error(label, err);
          return;
        }
        await new Promise((resolve) =>
          setTimeout(resolve, delay + Math.floor(Math.random() * 50)),
        );
        delay *= 2;
      }
    }
  }
}

class LangChainResponseHandler {
  private controller: AbortController | null = null;
  private messageText = '';
  private finalized = false;
  private aborted = false;
  private currentIndicator?: IndicatorState;
  private indicatorCleared = false;
  private pendingUpdateTimer: ReturnType<typeof setTimeout> | null = null;
  private lastUpdatePromise: Promise<void> = Promise.resolve();
  private readonly updateIntervalMs = 200;
  private disposed = false;
  private readonly toolsResolver: () => AgentTool[];
  private readonly persistMemories?: (responseText: string) => Promise<void>;

  constructor(
    private readonly model: BaseChatModel,
    private readonly chatClient: StreamChat,
    private readonly channel: Channel,
    private readonly message: MessageResponse,
    private readonly messages: CoreMessage[],
    private readonly sendEvent: (event: Record<string, unknown>) => Promise<void>,
    toolsResolver: () => AgentTool[],
    persistMemories?: (responseText: string) => Promise<void>,
  ) {
    this.chatClient.on('ai_indicator.stop', this.handleStopGenerating);
    this.toolsResolver = toolsResolver;
    this.persistMemories = persistMemories;
  }

  run = async () => {
    this.controller = new AbortController();

    try {
      const tools = this.buildAgentTools();
      await this.generateResponse(tools);
    } catch (error) {
      if (this.aborted) {
        await this.finalizeMessage();
      } else {
        await this.handleError(error as Error);
      }
    } finally {
      this.controller = null;
    }
  };

  private async generateResponse(tools: ToolExecutor[]): Promise<void> {
    const lcMessages = this.messages.map((message) =>
      message.role === 'system'
        ? new SystemMessage(message.content)
        : message.role === 'assistant'
          ? new AIMessage(message.content)
          : new HumanMessage(message.content),
    );

    let boundModel: BaseChatModel = this.model;
    if (tools.length) {
      const lcTools = tools.map((tool) => tool.langChainTool);
      boundModel = boundModel.bindTools(lcTools);
    }

    while (!this.aborted) {
      const aiMessage = await this.streamModel(boundModel, lcMessages);
      if (!aiMessage) {
        break;
      }

      const toolCalls = this.parseToolCalls(aiMessage);
      if (!toolCalls.length) {
        await this.finalizeMessage();
        if (this.persistMemories && this.messageText.trim().length) {
          await this.persistMemories(this.messageText.trim());
        }
        break;
      }

      await this.handleToolCalls(toolCalls, tools, lcMessages);
      this.messageText = '';
    }
  }

  private async streamModel(
    model: BaseChatModel,
    lcMessages: BaseMessage[],
  ): Promise<AIMessage | null> {
    this.messageText = '';
    let finalChunk: AIMessageChunk | null = null;
    const stream = await this.createModelStream(model, lcMessages);

    for await (const chunk of stream) {
      const aiChunk = chunk as AIMessageChunk;
      finalChunk = finalChunk ? finalChunk.concat(aiChunk) : aiChunk;
      const delta = extractTextFromChunk(aiChunk);
      if (delta) {
        await this.updateIndicator('AI_STATE_GENERATING');
        this.messageText += delta;
        this.schedulePartialUpdate();
      }
    }

    if (!finalChunk) {
      return null;
    }

    const aiMessage = chunkToAiMessage(finalChunk);
    lcMessages.push(aiMessage);
    return aiMessage;
  }

  private async createModelStream(
    model: BaseChatModel,
    lcMessages: BaseMessage[],
  ): Promise<AsyncIterable<AIMessageChunk>> {
    const streamMessages = (model as BaseChatModel & {
      streamMessages?: (
        messages: BaseMessage[],
        options?: Record<string, unknown>,
      ) => Promise<AsyncIterable<AIMessageChunk>>;
    }).streamMessages;

    if (typeof streamMessages === 'function') {
      return streamMessages.call(model, lcMessages, {
        signal: this.controller?.signal ?? undefined,
      });
    }

    const stream = (await model.stream(lcMessages, {
      signal: this.controller?.signal ?? undefined,
    })) as AsyncIterable<AIMessageChunk>;
    return stream;
  }

  private async handleToolCalls(
    toolCalls: ToolCall[],
    tools: ToolExecutor[],
    lcMessages: BaseMessage[],
  ) {
    for (const call of toolCalls) {
      const tool = tools.find((toolDef) => toolDef.name === call.name);
      if (!tool) {
        continue;
      }
      if (tool.showExternalSourcesIndicator) {
        await this.updateIndicator('AI_STATE_EXTERNAL_SOURCES');
      }
      const result = await tool.execute(call.args, {
        channel: this.channel,
        message: this.message,
        sendEvent: (event) => this.sendEvent(event),
      });
      const output = typeof result === 'string' ? result : JSON.stringify(result);
      const toolMessage = new ToolMessage(output, call.id ?? call.name ?? tool.name);
      lcMessages.push(toolMessage);
    }
  }

  private parseToolCalls(aiMessage: AIMessage): ToolCall[] {
    if (!aiMessage.tool_calls?.length) {
      return [];
    }
    return aiMessage.tool_calls.map((call) => ({
      name: call.name,
      id: call.id,
      args: call.args,
    }));
  }

  private buildAgentTools(): ToolExecutor[] {
    const tools = this.toolsResolver();
    if (!tools.length) {
      return [];
    }
    return tools.map((tool) => ({
      name: tool.name,
      langChainTool: new DynamicStructuredTool({
        name: tool.name,
        description: tool.description,
        schema: tool.parameters,
        func: async () => '',
      }),
      showExternalSourcesIndicator: tool.showExternalSourcesIndicator !== false,
      execute: tool.execute,
    }));
  }

  dispose = async () => {
    if (this.disposed) return;
    this.disposed = true;
    this.chatClient.off('ai_indicator.stop', this.handleStopGenerating);
    if (this.pendingUpdateTimer) {
      clearTimeout(this.pendingUpdateTimer);
      this.pendingUpdateTimer = null;
    }
  };

  private handleStopGenerating = async (event: Event) => {
    const messageId = (event as unknown as { message_id?: string })?.message_id;
    if (messageId && messageId !== this.message.id) {
      return;
    }

    this.aborted = true;
    try {
      this.controller?.abort();
    } catch (e) {
      // no-op
    }
  };

  private schedulePartialUpdate() {
    if (this.finalized) return;
    if (this.pendingUpdateTimer) return;
    this.pendingUpdateTimer = setTimeout(() => {
      this.pendingUpdateTimer = null;
      void this.flushPartialUpdate();
    }, this.updateIntervalMs);
  }

  private async flushPartialUpdate() {
    if (this.finalized) return;
    const text = this.messageText;
    const id = this.message.id;
    this.lastUpdatePromise = this.lastUpdatePromise.then(() =>
      this.chatClient
        .partialUpdateMessage(id, {
          set: { text, generating: true } as any,
        })
        .then(() => undefined),
    );
    await this.lastUpdatePromise;
  }

  private async updateIndicator(state: IndicatorState) {
    if (this.currentIndicator === state) return;
    this.currentIndicator = state;
    this.indicatorCleared = false;
    await this.sendEvent({
      type: 'ai_indicator.update',
      ai_state: state,
      cid: this.message.cid,
      message_id: this.message.id,
    });
  }

  private async clearIndicator() {
    if (this.indicatorCleared) return;
    this.currentIndicator = undefined;
    this.indicatorCleared = true;
    await this.sendEvent({
      type: 'ai_indicator.clear',
      cid: this.message.cid,
      message_id: this.message.id,
    });
  }

  private async finalizeMessage() {
    if (this.finalized) return;
    this.finalized = true;
    if (this.pendingUpdateTimer) {
      clearTimeout(this.pendingUpdateTimer);
      this.pendingUpdateTimer = null;
    }
    await this.lastUpdatePromise.catch(() => Promise.resolve());
    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: { text: this.messageText, generating: false } as any,
    });
    await this.clearIndicator();
  }

  private async handleError(error: Error) {
    this.finalized = true;
    await this.sendEvent({
      type: 'ai_indicator.update',
      ai_state: 'AI_STATE_ERROR',
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: {
        text: error.message ?? 'Error generating the message',
        generating: false,
      } as any,
    });
  }
}

type ToolExecutor = {
  name: string;
  langChainTool: LangChainTool;
  execute: AgentTool['execute'];
  showExternalSourcesIndicator: boolean;
};

const extractTextFromChunk = (chunk: AIMessageChunk): string => {
  if (!chunk.content) {
    return '';
  }
  if (typeof chunk.content === 'string') {
    return chunk.content;
  }
  if (Array.isArray(chunk.content)) {
    return chunk.content
      .filter((part) => part.type === 'text' && typeof part.text === 'string')
      .map((part) => part.text)
      .join('');
  }
  return '';
};

const chunkToAiMessage = (chunk: BaseMessage): AIMessage => {
  if (isAIMessage(chunk)) {
    return chunk;
  }
  const aiChunk = chunk as AIMessageChunk;
  return new AIMessage({
    content: aiChunk.content,
    name: aiChunk.name,
    id: aiChunk.id,
    additional_kwargs: { ...(aiChunk.additional_kwargs ?? {}) },
    response_metadata: { ...(aiChunk.response_metadata ?? {}) },
    tool_calls: aiChunk.tool_calls,
    invalid_tool_calls: aiChunk.invalid_tool_calls,
    usage_metadata: aiChunk.usage_metadata,
  });
};

const jsonSchemaToZod = (schema?: JsonSchemaDefinition): ZodTypeAny => {
  if (!schema) {
    return z.object({}).passthrough();
  }

  if (
    Array.isArray(schema.enum) &&
    schema.enum.length > 0 &&
    schema.enum.every((value) => typeof value === 'string')
  ) {
    const enumValues = schema.enum as string[];
    const enumSchema =
      enumValues.length === 1
        ? z.literal(enumValues[0])
        : z.enum(enumValues as [string, ...string[]]);
    return applyDescription(enumSchema, schema.description);
  }

  const inferredType = schema.type ?? (schema.properties ? 'object' : undefined);

  switch (inferredType) {
    case 'string':
      return applyDescription(z.string(), schema.description);
    case 'number':
      return applyDescription(z.number(), schema.description);
    case 'integer':
      return applyDescription(z.number().int(), schema.description);
    case 'boolean':
      return applyDescription(z.boolean(), schema.description);
    case 'array': {
      const itemSchemas = Array.isArray(schema.items)
        ? schema.items
        : schema.items
          ? [schema.items]
          : [];
      const firstItemSchema = itemSchemas[0] as JsonSchemaDefinition | undefined;
      const itemZod = firstItemSchema
        ? jsonSchemaToZod(firstItemSchema)
        : z.any();
      return applyDescription(z.array(itemZod), schema.description);
    }
    case 'object':
    default: {
      const properties = schema.properties ?? {};
      const required = new Set(schema.required ?? []);
      const shape: Record<string, ZodTypeAny> = {};
      for (const [key, propertySchema] of Object.entries(properties)) {
        let fieldSchema = jsonSchemaToZod(propertySchema);
        if (!required.has(key)) {
          fieldSchema = fieldSchema.optional();
        }
        shape[key] = fieldSchema;
      }

      let objectSchema: any = z.object(shape);
      const additional = schema.additionalProperties;
      if (typeof additional === 'object') {
        objectSchema = objectSchema.catchall(jsonSchemaToZod(additional));
      } else if (additional === false) {
        objectSchema = objectSchema.strict();
      } else {
        objectSchema = objectSchema.passthrough();
      }

      return applyDescription(objectSchema as ZodTypeAny, schema.description);
    }
  }
};

const applyDescription = <T extends ZodTypeAny>(
  schema: T,
  description?: string,
): T => {
  if (description) {
    return schema.describe(description) as T;
  }
  return schema;
};
